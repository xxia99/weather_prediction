# (full updated CB_encoder_stage2.py)
# -- snip: imports and config unchanged except as noted --

import os
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import xarray as xr
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from demo_evaluate import load_truth_wide, per_county_rmse_mean

# ---------------- config ----------------
SPLIT_DIR = "data/"
OUT_DIR   = "results_109/"
os.makedirs(OUT_DIR, exist_ok=True)

# core training hyperparameters
SEQ_LEN       = 24
BATCH_SIZE    = 64
EPOCHS        = 40
LR            = 5e-4
WEIGHT_DECAY  = 5e-5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# temporal dropout (kept)
TEMPORAL_DROP_PROB = 0.0

# === PatchTST encoder hyperparams ===
PATCH_LEN      = 2
PATCH_STRIDE   = None
PATCH_D_MODEL  = 64
PATCH_NHEAD    = 4
PATCH_NLAYERS  = 2
PATCH_DROPOUT  = 0.05
PATCH_FF       = 4 * PATCH_D_MODEL

# --------- StemGNN hyperparams ----------
STEM_D_MODEL  = 64
STEM_K_ORDER  = 2
STEM_NBLOCKS  = 2
STEM_D_EMB    = 32
STEM_DROPOUT  = 0.03

# === Pretrained encoder options (unchanged) ===
PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_validation.pt"
ENCODER_LR              = 3e-4
FREEZE_ENCODER          = False

# === Chronos-Bolt PEFT encoder (unchanged) ===
USE_CB_ENCODER          = True
CB_ENCODER_CKPT         = "results_lora_109/pretrained_cb_encoder_peft_independent_H24.pt"

# === Gating classifier from pretraining (upgraded) ===
USE_CLASSIFIER_GATING   = True
CLS_FROM_CKPT           = CB_ENCODER_CKPT
GATING_MODE             = "soft"      # "soft" (weight loss by p) or "hard" (mask by threshold) during training
STOP_GRAD_GATING        = True        # classifier is frozen; no grads to it

# NEW: classifier pooling & spatial mixing
CLASSIFIER_POOL         = "attn"      # "attn" (temporal attention) or "last" (last-step features)
CLASSIFIER_SPATIAL_MIX  = True        # light cross-node mixing per horizon inside classifier

# NEW: threshold auto-tuning on validation
AUTO_TUNE_THRESHOLD     = True
ZERO2NONZERO_TARGET     = 0.20        # cap FP-rate on true zeros (Zero→Nonzero)
THRESH_GRID             = np.linspace(0.05, 0.95, 19).tolist()

# default threshold (overwritten if AUTO_TUNE_THRESHOLD=True)
CLS_THRESHOLD           = 0.5

# === Day split for in-train validation ===
TRAIN_LAST_DAY = 80

# === Event-stratified validation (unchanged) ===
EVENT_METHOD   = "mean"
STORM_TOP_Q    = 0.80
CALM_BOTTOM_Q  = 0.50
CALM_N         = 10
BEST_BY        = "overall"

# ---------------- utils (unchanged) ----------------
def open_ds(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path, engine="netcdf4")

def get_all_features(ds):
    if "feature" not in ds.dims:
        raise RuntimeError("Dataset missing 'feature' dim for weather variables.")
    return list(map(str, ds.feature.values))

def zfit(arr):
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def zapply(arr, mu, sd):
    return (arr - mu) / sd

def build_windows(X_loc, y_loc, seq_len, horizon):
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return np.empty((0, seq_len, X_loc.shape[1]), dtype=float), np.empty((0, horizon), dtype=float)
    Xw, Yw = [], []
    for i in range(N):
        Xw.append(X_loc[i:i+seq_len])
        Yw.append(y_loc[i+seq_len:i+seq_len+horizon])
    return np.asarray(Xw, dtype=float), np.asarray(Yw, dtype=float)

# -------------------- PatchTST encoder --------------------
class PatchTSTEncoder(nn.Module):
    def __init__(self, in_channels, patch_len=PATCH_LEN, stride=PATCH_STRIDE,
                 d_model=PATCH_D_MODEL, nhead=PATCH_NHEAD, nlayers=PATCH_NLAYERS,
                 dropout=PATCH_DROPOUT, ff=PATCH_FF):
        super().__init__()
        self.in_channels = in_channels
        self.patch_len   = patch_len
        self.stride      = patch_len if stride is None else stride
        self.d_model     = d_model

        self.proj = nn.Linear(in_channels * self.patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, F = x.shape
        P, S = self.patch_len, self.stride
        assert F == self.in_channels, f"PatchTST got F={F}, expected {self.in_channels}"

        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)
        L = (T + S - 1) // S
        T_pad = (L - 1) * S + P
        if T_pad < T:
            L += 1
            T_pad = (L - 1) * S + P
        if T_pad > T:
            pad = torch.zeros(B * N, T_pad - T, F, device=x.device, dtype=x.dtype)
            x_bn = torch.cat([x_bn, pad], dim=1)

        x_unf = x_bn.unfold(dimension=1, size=P, step=S)          # (B*N, L, P, F)
        x_unf = x_unf.contiguous().view(B * N, L, P * F)          # (B*N, L, P*F)

        tokens = self.proj(x_unf)
        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)

        tokens_rep = tokens.repeat_interleave(P, dim=1)            # (B*N, L*P, d_model)
        steps = tokens_rep[:, :T, :]                               # (B*N, T, d_model)
        h = steps.view(B, N, T, self.d_model).permute(0, 2, 1, 3)
        return h

# -------------------- Chronos-Bolt PEFT encoder (unchanged) --------------------
def _discover_lora_targets(model: nn.Module):
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    candidates = [
        "q_proj","k_proj","v_proj","o_proj",
        "query_key_value","out_proj","in_proj",
        "to_q","to_k","to_v","to_out","proj_in","proj_out",
        "q","k","v","o","wi","wo","wi_0","wi_1",
        "Wqkv","Wo","Wq","Wk","Wv",
    ]
    present = [s for s in candidates if any(s in n for n in linear_names)]
    if not present:
        attn_linear = [n for n in linear_names if ("attn" in n.lower() or "attention" in n.lower())]
        if attn_linear:
            present = list({n.split(".")[-1] for n in attn_linear})
    if not present:
        enc_linear = [n for n in linear_names if (".encoder." in n or "encoder.layers" in n or ".layers." in n)]
        if enc_linear:
            present = list({n.split(".")[-1] for n in enc_linear})
    if not present:
        present = ["linear"]
    seen, final = set(), []
    for s in present:
        if s not in seen:
            seen.add(s)
            final.append(s)
    return final

class ChronosBoltEncoderPEFT(nn.Module):
    def __init__(self, in_channels, ckpt_path):
        super().__init__()
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"CB encoder ckpt not found: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu")
        cb_model_name = ck.get("cb_model_name", "amazon/chronos-bolt-mini")
        d_out         = int(ck.get("d_out", 64))
        self.in_channels = int(ck.get("in_channels", in_channels))
        self.d_model = d_out

        try:
            from transformers import AutoModel, AutoConfig
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise ImportError(
                "This script requires 'transformers' and 'peft' when USE_CB_ENCODER=True.\n"
                "Install: pip install transformers peft accelerate safetensors\n"
                f"Original import error: {e}"
            )

        config  = AutoConfig.from_pretrained(cb_model_name, trust_remote_code=True)
        backbone = AutoModel.from_pretrained(cb_model_name, trust_remote_code=True, output_hidden_states=True)

        lcfg = LoraConfig(
            r=int(ck.get("lora_r", 8)),
            lora_alpha=int(ck.get("lora_alpha", 16)),
            lora_dropout=float(ck.get("lora_dropout", 0.05)),
            target_modules=_discover_lora_targets(backbone),
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        backbone = get_peft_model(backbone, lcfg)

        hidden_dim = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if hidden_dim is None:
            raise RuntimeError("Cannot determine Chronos hidden size from config.")
        self.embed_proj = nn.Linear(self.in_channels, hidden_dim)
        self.post_proj  = nn.Linear(hidden_dim, self.d_model)

        self.backbone = backbone
        self.load_state_dict(ck["state_dict"], strict=True)  # load encoder weights
        print(f"[CB Encoder] Loaded PEFT encoder from: {ckpt_path} | d_out={self.d_model}")

    def forward(self, x):
        B, T, N, F = x.shape
        assert F == self.in_channels, f"CB encoder expected F={self.in_channels}, got {F}"

        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        emb  = self.embed_proj(x_bn)
        attn_mask = torch.ones(B * N, T, dtype=torch.long, device=emb.device)

        if hasattr(self.backbone, "encoder") and callable(self.backbone.encoder):
            out = self.backbone.encoder(
                inputs_embeds=emb, attention_mask=attn_mask, output_hidden_states=True
            )
            h_bn = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        else:
            out = self.backbone(
                inputs_embeds=emb, attention_mask=attn_mask, output_hidden_states=True
            )
            h_bn = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        if h_bn.size(1) > T:
            h_bn = h_bn[:, :T, :]
        elif h_bn.size(1) < T:
            pad_len = T - h_bn.size(1)
            pad = torch.zeros(h_bn.size(0), pad_len, h_bn.size(2), device=h_bn.device, dtype=h_bn.dtype)
            h_bn = torch.cat([h_bn, pad], dim=1)

        h_bn = self.post_proj(h_bn)
        h = h_bn.view(B, N, T, self.d_model).permute(0, 2, 1, 3)
        return h

# -------- NEW: Tiny temporal attention + optional spatial mixing for classifier --------
class TemporalAttentionPool(nn.Module):
    """
    One-query-per-horizon temporal attention.
    Input h: (B, T, N, D) -> returns ctx: (B, N, H, D)
    """
    def __init__(self, d_model, horizon):
        super().__init__()
        self.d = d_model
        self.horizon = horizon
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Q  = nn.Parameter(torch.randn(horizon, d_model) / math.sqrt(d_model))

    def forward(self, h):
        # h: (B,T,N,D)
        B, T, N, D = h.shape
        k = self.Wk(h)                           # (B,T,N,D)
        v = self.Wv(h)                           # (B,T,N,D)
        # scores: (B,N,H,T)
        scores = torch.einsum("hd,btnd->bnht", self.Q, k) / math.sqrt(D)
        attn = torch.softmax(scores, dim=-1)     # (B,N,H,T)
        # ctx: (B,N,H,D)
        ctx = torch.einsum("bnht,btnd->bnhd", attn, v)
        return ctx

class SpatialMix(nn.Module):
    """
    Light cross-node mixing per horizon via self-attention across nodes.
    Input ctx: (B, N, H, D) -> returns mixed: (B, N, H, D)
    """
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, ctx):
        B, N, H, D = ctx.shape
        x = ctx.permute(0, 2, 1, 3).reshape(B * H, N, D)  # (B*H, N, D)
        scores = torch.einsum("bnd,bmd->bnm", x, x) / math.sqrt(D)  # (B*H, N, N)
        A = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("bnm,bmd->bnd", A, x)                  # (B*H, N, D)
        mixed = mixed.view(B, H, N, D).permute(0, 2, 1, 3)          # (B,N,H,D)
        out = self.gamma * self.dropout(mixed) + (1.0 - self.gamma) * ctx
        return out

# -------- NEW: Outage classifier module --------
class CBOutageClassifier(nn.Module):
    """
    LoRA CB encoder + classification head for outage(>0) over H steps.
    Pooling: 'attn' (temporal attention) or 'last'
    Optional spatial mixing across nodes per horizon.
    """
    def __init__(self, in_channels, horizon, ckpt_path,
                 pooling=CLASSIFIER_POOL, spatial_mix=CLASSIFIER_SPATIAL_MIX):
        super().__init__()
        assert pooling in ("attn", "last")
        self.pooling = pooling
        self.spatial_mix_enabled = spatial_mix

        self.encoder = ChronosBoltEncoderPEFT(in_channels=in_channels, ckpt_path=ckpt_path)
        d = self.encoder.d_model

        if self.pooling == "attn":
            self.temporal_pool = TemporalAttentionPool(d_model=d, horizon=horizon)
            if spatial_mix:
                self.spatial = SpatialMix(d_model=d, dropout=0.05)
            self.cls_head_attn = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d, 1)
            )
        else:
            self.temporal_pool = None
            self.spatial = SpatialMix(d_model=d, dropout=0.05) if spatial_mix else None
            self.cls_head = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(2 * d, horizon)
            )

        # Try loading heads from ckpt if available (shape must match)
        ck = torch.load(ckpt_path, map_location="cpu")
        if self.pooling == "attn":
            key = "cls_head_attn_state_dict"
            if key in ck:
                try:
                    self.cls_head_attn.load_state_dict(ck[key], strict=True)
                    print(f"[Classifier] Loaded attention head from {ckpt_path}")
                except Exception as e:
                    print(f"[Classifier] Warning: failed to load attention head: {e}")
            else:
                print(f"[Classifier] No '{key}' in {ckpt_path}; using randomly initialized attention head.")
        else:
            key = "cls_head_state_dict"
            if key in ck:
                try:
                    self.cls_head.load_state_dict(ck[key], strict=True)
                    print(f"[Classifier] Loaded classifier head from {ckpt_path}")
                except Exception as e:
                    print(f"[Classifier] Warning: failed to load classifier head: {e}")
            else:
                print(f"[Classifier] No '{key}' found in {ckpt_path}; using randomly initialized head.")

        # freeze by default (two-stage)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B,T,N,F_full) scaled inputs; classifier pulls WEATHER-ONLY tail.
        returns logits: (B,H,N)
        """
        needF = self.encoder.in_channels
        x_weather = x[..., -needF:] if x.shape[-1] != needF else x
        h = self.encoder(x_weather)   # (B,T,N,D)

        if self.pooling == "attn":
            ctx = self.temporal_pool(h)  # (B,N,H,D)
            if self.spatial_mix_enabled:
                ctx = self.spatial(ctx)  # (B,N,H,D)
            logits = self.cls_head_attn(ctx).squeeze(-1)  # (B,N,H)
            return logits.permute(0, 2, 1)                # (B,H,N)
        else:
            last = h[:, -1]  # (B,N,D)
            if self.spatial_mix_enabled:
                ctx = last[:, :, None, :]  # (B,N,1,D)
                ctx = self.spatial(ctx)    # (B,N,1,D)
                last = ctx[:, :, 0, :]     # (B,N,D)
            logits = self.cls_head(last)   # (B,N,H)
            return logits.permute(0, 2, 1) # (B,H,N)

# -------------------- StemGNN blocks (unchanged) --------------------
class _TemporalConv(nn.Module):
    def __init__(self, c, dropout=0.0, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(c, c, kernel_size=kernel_size, padding=padding, groups=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, C = x.shape
        x_bn = x.reshape(B * N, T, C).permute(0, 2, 1)
        y = self.conv(x_bn)
        y = self.act(y)
        y = self.dropout(y)
        y = y.permute(0, 2, 1).reshape(B, T, N, C)
        return y

class _AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes, d_emb):
        super().__init__()
        self.E = nn.Parameter(torch.randn(num_nodes, d_emb))
    def forward(self):
        E = self.E
        A = torch.relu(E @ E.t())
        A = torch.softmax(A, dim=1)
        return A

class _GraphDiffusion(nn.Module):
    def __init__(self, k_order, d_model, dropout=0.0):
        super().__init__()
        self.k_order = k_order
        self.alpha = nn.Parameter(torch.ones(k_order + 1))
        self.dropout = nn.Dropout(dropout)
        self.mix = nn.Linear(d_model, d_model)
    def forward(self, x, A):
        if self.k_order <= 0:
            return self.dropout(self.mix(x))
        B, T, N, C = x.shape
        I = torch.eye(N, device=x.device, dtype=x.dtype)
        A_pows = [I]
        Ak = A
        for _ in range(1, self.k_order + 1):
            A_pows.append(Ak)
            Ak = Ak @ A
        agg = 0.0
        for k, Ak in enumerate(A_pows):
            y = torch.einsum('ij,btjc->btic', Ak, x)
            agg = agg + self.alpha[k] * y
        out = self.dropout(self.mix(agg))
        return out

class _StemBlock(nn.Module):
    def __init__(self, d_in, d_model, k_order, num_nodes, d_emb, dropout=0.0):
        super().__init__()
        self.in_proj  = nn.Linear(d_in, d_model)
        self.tconv    = _TemporalConv(d_model, dropout=dropout)
        self.adaptA   = _AdaptiveAdjacency(num_nodes, d_emb)
        self.gdiff    = _GraphDiffusion(k_order, d_model, dropout=dropout)
        self.out_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        h0 = self.in_proj(x)
        ht = self.tconv(h0)
        A  = self.adaptA()
        hg = self.gdiff(ht, A)
        h  = self.out_norm(h0 + hg)
        return h

# -------------------- StemGNN Forecaster (unchanged API) --------------------
class STGNNForecaster(nn.Module):
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels

        if USE_CB_ENCODER and os.path.exists(CB_ENCODER_CKPT):
            self.patch = ChronosBoltEncoderPEFT(in_channels=in_channels, ckpt_path=CB_ENCODER_CKPT)
            print("[Encoder] Using Chronos-Bolt PEFT encoder.")
        else:
            self.patch = PatchTSTEncoder(
                in_channels=in_channels,
                patch_len=PATCH_LEN,
                stride=PATCH_STRIDE,
                d_model=PATCH_D_MODEL,
                nhead=PATCH_NHEAD,
                nlayers=PATCH_NLAYERS,
                dropout=PATCH_DROPOUT,
                ff=PATCH_FF
            )
            if PRETRAINED_ENCODER_PATH is not None and os.path.exists(PRETRAINED_ENCODER_PATH):
                try:
                    ckpt = torch.load(PRETRAINED_ENCODER_PATH, map_location="cpu")
                    state = ckpt.get("state_dict", ckpt)
                    self.patch.load_state_dict(state, strict=False)
                    print(f"[Encoder] Loaded pretrained PatchTST from: {PRETRAINED_ENCODER_PATH}")
                except Exception as e:
                    print(f"[Encoder] Warning: failed to load pretrained PatchTST: {e}")

        if FREEZE_ENCODER:
            for p in self.patch.parameters():
                p.requires_grad = False
            print("[Encoder] Frozen (no fine-tuning).")

        enc_out_dim = getattr(self.patch, "d_model", PATCH_D_MODEL)
        blocks = []
        d_in = enc_out_dim
        for _ in range(STEM_NBLOCKS):
            blocks.append(
                _StemBlock(
                    d_in=d_in,
                    d_model=STEM_D_MODEL,
                    k_order=STEM_K_ORDER,
                    num_nodes=num_nodes,
                    d_emb=STEM_D_EMB,
                    dropout=STEM_DROPOUT,
                )
            )
            d_in = STEM_D_MODEL
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Linear(STEM_D_MODEL, horizon)

    def forward(self, x):
        B, T, N, F = x.shape
        assert N == self.num_nodes and F == self.in_channels, \
            f"StemGNN input shape mismatch: got N={N},F={F}, expected N={self.num_nodes},F={self.in_channels}"

        if USE_CB_ENCODER and isinstance(self.patch, ChronosBoltEncoderPEFT):
            needF = self.patch.in_channels  # weather-only encoder
            x_patch = x[..., -needF:] if x.shape[-1] != needF else x
        else:
            x_patch = x
        h = self.patch(x_patch)

        for blk in self.blocks:
            h = blk(h)
        last = h[:, -1]  # (B,N,D)
        y = self.readout(last)  # (B,N,H)
        return y

# ---------------- helpers for day split & windows (unchanged) ----------------
def _compute_day_indices(ts: np.ndarray):
    ts_pd = pd.to_datetime(ts)
    days_norm = ts_pd.normalize()
    uniq = days_norm.unique()
    uniq = pd.DatetimeIndex(sorted(uniq))
    day_map = {d: i+1 for i, d in enumerate(uniq)}
    days = np.array([day_map[d] for d in days_norm], dtype=int)
    day_end_idx = []
    for d in uniq:
        mask = (days_norm == d)
        idxs = np.where(mask)[0]
        day_end_idx.append(int(idxs.max()))
    return days, list(uniq), day_end_idx

def _fit_scalers_train_only(y, w, days, train_last_day):
    mask = days <= train_last_day
    y_tr = y[mask, :].reshape(-1, 1)
    w_tr = w[mask, :, :].reshape(-1, w.shape[2])
    y_mu, y_sd = zfit(y_tr)
    w_mu, w_sd = zfit(w_tr)
    return {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}

def prepare_training_split(ds_train, horizon, train_last_day=80):
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, uniq_days, day_end_idx = _compute_day_indices(ts)
    T, N, F = w.shape
    Din = 1 + F

    scalers = _fit_scalers_train_only(y, w, days, train_last_day)

    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)  # (T, N, Din)

    N_total = T - SEQ_LEN - horizon + 1
    if N_total <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    day_last_idx = day_end_idx[train_last_day - 1]
    N_train = max(0, day_last_idx - SEQ_LEN - horizon + 1)

    X_list, Y_list = [], []
    for node in range(N):
        x_loc_full = X_all[:, node, :]
        Xw, Yw = build_windows(x_loc_full, y_sc[:, node], SEQ_LEN, horizon)
        if len(Xw):
            X_list.append(Xw[:N_train][:, :, None, :])
            Y_list.append(Yw[:N_train][:, None, :])

    if not X_list:
        raise RuntimeError("No training windows after split; check SEQ_LEN/horizon/TRAIN_LAST_DAY.")
    X_tr = np.concatenate(X_list, axis=0)  # (nW*N, SEQ_LEN, 1, Din)
    Y_tr = np.concatenate(Y_list, axis=0)  # (nW*N, 1, H)

    aux = {
        "X_all_scaled": X_all,
        "y_scaled": y_sc,
        "timestamps": ts,
        "days": days,
        "day_end_idx": day_end_idx
    }
    return X_tr, Y_tr, Din, scalers, aux

# ---------------- helper: truth melt + zero/nonzero ratios ----------------
def _truth_long(df_truth_wide: pd.DataFrame) -> pd.DataFrame:
    if df_truth_wide is None or df_truth_wide.empty:
        return pd.DataFrame(columns=["timestamp","location","y"])
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]
    df_long = df_truth_wide.melt(id_vars="timestamp", value_vars=locs,
                                 var_name="location", value_name="y").copy()
    df_long["timestamp"] = pd.to_datetime(df_long["timestamp"])
    df_long["location"]  = df_long["location"].astype(str)
    return df_long

def _ratio_metrics(df_truth_wide: pd.DataFrame, df_probs_long: pd.DataFrame, thr: float):
    """
    df_probs_long columns: [timestamp, location, p] with p in [0,1]
    Returns (Zero->Nonzero, Nonzero->Zero, counts_zero, counts_pos)
    """
    if df_truth_wide.empty or df_probs_long.empty:
        return float("nan"), float("nan"), 0, 0
    df_t = _truth_long(df_truth_wide)
    df_p = df_probs_long.copy()
    df_p["timestamp"] = pd.to_datetime(df_p["timestamp"])
    df_p["location"]  = df_p["location"].astype(str)
    df = df_t.merge(df_p, on=["timestamp","location"], how="inner")
    if df.empty:
        return float("nan"), float("nan"), 0, 0
    zero_mask = df["y"] <= 0.0
    pos_mask  = df["y"] > 0.0
    pred_nonzero = df["p"].values >= float(thr)
    zeros_total = int(zero_mask.sum())
    pos_total   = int(pos_mask.sum())
    z2nz = float((zero_mask & pred_nonzero).sum() / zeros_total) if zeros_total > 0 else float("nan")
    nz2z = float((pos_mask & (~pred_nonzero)).sum() / pos_total) if pos_total > 0 else float("nan")
    return z2nz, nz2z, zeros_total, pos_total

# ---------------- validation probs with spatial context ----------------
@torch.no_grad()
def predict_val_probs(ds_train, classifier, aux, horizon, train_last_day=80):
    """
    Compute classifier probabilities for ALL nodes using spatial context, per validation day.
    Returns a long df with [timestamp, location, p] (p=Prob(Y>0)).
    """
    if classifier is None:
        return pd.DataFrame(columns=["timestamp","location","p"])
    classifier.eval()

    X_all = aux["X_all_scaled"]           # (T, N, Din)
    ts    = aux["timestamps"]
    days  = aux["days"]
    day_end_idx = aux["day_end_idx"]
    N = X_all.shape[1]
    Din = X_all.shape[2]
    locs = list(map(str, ds_train.location.values))

    rows = []
    last_day = int(days.max())
    for d in range(train_last_day, last_day):
        end_idx = day_end_idx[d - 1]
        start_ctx = end_idx - (SEQ_LEN - 1)
        fut_start = end_idx + 1
        fut_end   = fut_start + horizon - 1
        if start_ctx < 0 or fut_end >= len(ts):
            continue
        ts_future = pd.to_datetime(ts[fut_start:fut_start+horizon])

        Xin_all = X_all[start_ctx:end_idx+1, :, :].reshape(1, SEQ_LEN, N, Din)  # (1,T,N,Din)
        xb_all  = torch.tensor(Xin_all, dtype=torch.float32, device=DEVICE)
        logits  = classifier(xb_all)[0].detach().cpu().numpy()  # (H,N_probs)
        p_all   = 1.0 / (1.0 + np.exp(-logits))                 # (H,N_probs)

        n_probs = p_all.shape[1]
        n_locs  = len(locs)
        if n_probs != n_locs:
            use_n = min(n_probs, n_locs)
            print(f"[Classifier] Warning: N_probs ({n_probs}) != N_locs ({n_locs}); aligning to first {use_n}.")
        else:
            use_n = n_locs

        for i, county in enumerate(locs[:use_n]):
            rows.append(pd.DataFrame({
                "timestamp": ts_future,
                "location": county,
                "p": p_all[:, i].astype(float)
            }))

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["timestamp","location","p"])

def truth_wide_for_val(ds_train, aux, horizon, train_last_day=80):
    ts    = aux["timestamps"]
    days  = aux["days"]
    day_end_idx = aux["day_end_idx"]

    fut_indices = []
    last_day = int(days.max())
    for d in range(train_last_day, last_day):
        end_idx = day_end_idx[d - 1]
        fut_start = end_idx + 1
        fut_end   = fut_start + horizon - 1
        if fut_end < len(ts):
            fut_indices.extend(list(range(fut_start, fut_start + horizon)))
    if not fut_indices:
        return pd.DataFrame(columns=["timestamp"])

    fut_indices = sorted(set(fut_indices))
    ts_fut = pd.to_datetime(ts[fut_indices])

    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    locs = list(map(str, ds_train.location.values))
    df = pd.DataFrame(y[fut_indices, :], columns=locs, index=ts_fut).reset_index().rename(columns={"index": "timestamp"})
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def _all_zero_rmse_for_truth(df_truth_wide: pd.DataFrame) -> float:
    if df_truth_wide.empty:
        return float("nan")
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]
    df_zero_long = pd.DataFrame({
        "timestamp": np.repeat(df_truth_wide["timestamp"].values, len(locs)),
        "location": np.tile(locs, len(df_truth_wide)),
        "pred": 0.0
    })
    return per_county_rmse_mean(df_truth_wide, df_zero_long)

# ---- event selection helpers (unchanged) ----
def _per_day_outage_score(ds_train, method: str = "mean"):
    ts = ds_train.timestamp.values
    days, _, _ = _compute_day_indices(ts)
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    df = pd.DataFrame({"day": days, "y": y.mean(axis=1)})
    if method.lower() == "max":
        g = df.groupby("day")["y"].max()
    else:
        g = df.groupby("day")["y"].mean()
    max_day = int(days.max())
    day_score = np.zeros(max_day, dtype=float)
    day_score[g.index.values - 1] = g.values
    return days, day_score

def _timestamps_for_days(ts: np.ndarray, select_days: set[int]) -> pd.DatetimeIndex:
    ts_pd = pd.to_datetime(ts)
    days, _, _ = _compute_day_indices(ts_pd)
    mask = np.array([ (d in select_days) for d in days ], dtype=bool)
    return pd.DatetimeIndex(ts_pd[mask])

def _filter_truth_pred_by_days(df_truth_wide: pd.DataFrame,
                               df_pred_long: pd.DataFrame,
                               ts_subset: pd.DatetimeIndex):
    df_truth_sel = df_truth_wide[df_truth_wide["timestamp"].isin(ts_subset)].copy()
    df_pred_sel  = df_pred_long[df_pred_long["timestamp"].isin(ts_subset)].copy()
    if not df_truth_sel.empty:
        df_truth_sel["timestamp"] = pd.to_datetime(df_truth_sel["timestamp"])
    if not df_pred_sel.empty:
        df_pred_sel["timestamp"] = pd.to_datetime(df_pred_sel["timestamp"])
    return df_truth_sel, df_pred_sel

def _event_rmse_blocks(ds_train, df_truth_val, df_pred_val,
                       train_last_day, method, storm_top_q, calm_bottom_q, calm_n, seed):
    rmse_overall = per_county_rmse_mean(df_truth_val, df_pred_val) if (not df_truth_val.empty and not df_pred_val.empty) else float("nan")
    zero_overall = _all_zero_rmse_for_truth(df_truth_val) if not df_truth_val.empty else float("nan")

    rng = np.random.default_rng(seed)
    ts = ds_train.timestamp.values
    days, day_score = _per_day_outage_score(ds_train, method=method)
    last_day = int(days.max())
    tgt_days = np.arange(train_last_day + 1, last_day + 1, dtype=int)
    if len(tgt_days) == 0:
        return rmse_overall, zero_overall, float("nan"), float("nan"), 0, float("nan"), float("nan"), 0

    tgt_scores = day_score[tgt_days - 1]
    storm_thr = np.quantile(tgt_scores, storm_top_q)
    calm_thr  = np.quantile(tgt_scores, calm_bottom_q)

    storm_days = set(tgt_days[tgt_scores >= storm_thr].tolist())
    calm_pool  = tgt_days[tgt_scores <= calm_thr].tolist()
    if len(calm_pool) > 0:
        k = int(min(calm_n, len(calm_pool)))
        calm_days = set(rng.choice(calm_pool, size=k, replace=False).tolist())
    else:
        calm_days = set()

    ts_storm = _timestamps_for_days(ts, storm_days)
    ts_calm  = _timestamps_for_days(ts, calm_days)

    if len(ts_calm) > 0:
        truth_calm, pred_calm = _filter_truth_pred_by_days(df_truth_val, df_pred_val, ts_calm)
        rmse_calm = per_county_rmse_mean(truth_calm, pred_calm)
        zero_calm = _all_zero_rmse_for_truth(truth_calm)
        n_calm = len(calm_days)
    else:
        rmse_calm, zero_calm, n_calm = float("nan"), float("nan"), 0

    if len(ts_storm) > 0:
        truth_storm, pred_storm = _filter_truth_pred_by_days(df_truth_val, df_pred_val, ts_storm)
        rmse_storm = per_county_rmse_mean(truth_storm, pred_storm)
        zero_storm = _all_zero_rmse_for_truth(truth_storm)
        n_storm = len(storm_days)
    else:
        rmse_storm, zero_storm, n_storm = float("nan"), float("nan"), 0

    return rmse_overall, zero_overall, rmse_calm, zero_calm, n_calm, rmse_storm, zero_storm, n_storm

# ---------------- threshold tuning ----------------
def tune_threshold(df_truth_val, df_probs_val, target_zero2nonzero=ZERO2NONZERO_TARGET, grid=THRESH_GRID):
    best_thr = None
    best_tuple = (float("inf"), float("inf"))  # (penalty, nz2z)
    best_metrics = (float("nan"), float("nan"))
    for thr in grid:
        z2nz, nz2z, zc, pc = _ratio_metrics(df_truth_val, df_probs_val, thr)
        if not np.isfinite(z2nz) or not np.isfinite(nz2z):
            continue
        penalty = max(0.0, z2nz - target_zero2nonzero)
        cand = (penalty, nz2z)
        if cand < best_tuple:
            best_tuple = cand
            best_thr = thr
            best_metrics = (z2nz, nz2z)
    if best_thr is None:
        best_thr = 0.5
        best_metrics = _ratio_metrics(df_truth_val, df_probs_val, best_thr)[:2]
    return float(best_thr), best_metrics

def print_cls_metrics(title, df_truth_val, df_probs_val, thr):
    z2nz, nz2z, zc, pc = _ratio_metrics(df_truth_val, df_probs_val, thr)
    if np.isfinite(z2nz) and np.isfinite(nz2z):
        print(f"[{title}] Classifier @ τ={thr:.2f}: Zero→Nonzero={z2nz:.4f} (all-zero=0.0000) | "
              f"Nonzero→Zero={nz2z:.4f} (all-zero=1.0000) "
              f"[zeros={zc}, positives={pc}]")
    else:
        print(f"[{title}] Classifier @ τ={thr:.2f}: Zero→Nonzero=nan | Nonzero→Zero=nan")

def make_val_callback_rolling(ds_train, scalers, horizon, aux, label,
                              train_last_day=80,
                              ts_future_test=None, df_truth_test=None, zero_rmse_test=float("nan"),
                              classifier=None, cls_threshold=0.5,
                              df_truth_val_cached=None, df_probs_val_cached=None):
    best = {"rmse": float("inf"), "epoch": -1}
    df_truth_val = df_truth_val_cached if df_truth_val_cached is not None else truth_wide_for_val(ds_train, aux, horizon, train_last_day)

    def _callback(model, epoch):
        df_pred_val = predict_val_roll(
            ds_train, model, scalers, horizon, aux, train_last_day,
            classifier=classifier, cls_threshold=cls_threshold
        )

        (rmse_overall, zero_overall,
         rmse_calm, zero_calm, n_calm,
         rmse_storm, zero_storm, n_storm) = _event_rmse_blocks(
            ds_train, df_truth_val, df_pred_val,
            train_last_day=train_last_day,
            method=EVENT_METHOD,
            storm_top_q=STORM_TOP_Q,
            calm_bottom_q=CALM_BOTTOM_Q,
            calm_n=CALM_N,
            seed=42
        )

        if np.isfinite(rmse_overall):
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE={rmse_overall:.4f} (all-zero={zero_overall:.4f})")
        else:
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE=nan (all-zero=nan)")

        if np.isfinite(rmse_calm):
            print(f"[EventEval {label}] Calm-heavy ({n_calm} days): RMSE={rmse_calm:.4f} (all-zero={zero_calm:.4f})")
        else:
            print(f"[EventEval {label}] Calm-heavy: RMSE=nan (all-zero=nan)")

        if np.isfinite(rmse_storm):
            print(f"[EventEval {label}] Storm-heavy ({n_storm} days): RMSE={rmse_storm:.4f} (all-zero={zero_storm:.4f})")
        else:
            print(f"[EventEval {label}] Storm-heavy: RMSE=nan (all-zero=nan)")

        if classifier is not None and df_probs_val_cached is not None:
            print_cls_metrics(f"Val {label} Classify", df_truth_val, df_probs_val_cached, cls_threshold)

        metric_map = {"overall": rmse_overall, "calm": rmse_calm, "storm": rmse_storm}
        sel = metric_map.get(str(BEST_BY).lower(), rmse_overall)
        sel_val = sel if np.isfinite(sel) else rmse_overall

        if np.isfinite(sel_val) and sel_val < best["rmse"]:
            best["rmse"] = sel_val
            best["epoch"] = epoch
            save_path = os.path.join(OUT_DIR, f"best_enc-dec_{label}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "label": label,
                    "rmse_select": sel_val,
                    "rmse_overall": rmse_overall,
                    "rmse_calm": rmse_calm,
                    "rmse_storm": rmse_storm,
                    "best_by": str(BEST_BY).lower(),
                    "model_state_dict": model.state_dict(),
                },
                save_path,
            )
            print(f"[Val {label}] New best ({BEST_BY}) RMSE {sel_val:.4f} at epoch {epoch}. Saved: {save_path}")

        if ts_future_test is not None and df_truth_test is not None and len(ts_future_test) > 0:
            df_pred_test = predict_per_county(ds_train, model, scalers, horizon, ts_future_test,
                                              classifier=classifier, cls_threshold=cls_threshold)
            rmse_test = per_county_rmse_mean(df_truth_test, df_pred_test)
            zero_rmse_test_local = _all_zero_rmse_for_truth(df_truth_test)
            print(f"[Val {label}] Epoch {epoch}: Testing RMSE={rmse_test:.4f} (all-zero RMSE={zero_rmse_test_local:.4f})")

    return _callback

# ---------------- validation roll (with spatial gating, SAFE alignment) ----------------
@torch.no_grad()
def predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day=80,
                     classifier=None, cls_threshold=0.5):
    model.eval()
    X_all = aux["X_all_scaled"]
    ts    = aux["timestamps"]
    days  = aux["days"]
    day_end_idx = aux["day_end_idx"]
    N = X_all.shape[1]
    Din = X_all.shape[2]
    locs = list(map(str, ds_train.location.values))
    n_locs = len(locs)

    rows = []
    last_day = int(days.max())
    for d in range(train_last_day, last_day):
        end_idx = day_end_idx[d - 1]
        start_ctx = end_idx - (SEQ_LEN - 1)
        fut_start = end_idx + 1
        fut_end   = fut_start + horizon - 1
        if start_ctx < 0 or fut_end >= len(ts):
            continue

        ts_future = pd.to_datetime(ts[fut_start:fut_start+horizon])

        # classifier once for all nodes (spatial context)
        if classifier is not None:
            Xin_all = X_all[start_ctx:end_idx+1, :, :].reshape(1, SEQ_LEN, N, Din)
            xb_all  = torch.tensor(Xin_all, dtype=torch.float32, device=DEVICE)
            logits_all = classifier(xb_all)[0].detach().cpu().numpy()  # (H,N_probs)
            p_all = 1.0 / (1.0 + np.exp(-logits_all))                  # (H,N_probs)
            n_probs = p_all.shape[1]
            if n_probs != n_locs:
                print(f"[Classifier] Warning: N_probs ({n_probs}) != N_locs ({n_locs}); gating aligned by index where available.")
        else:
            p_all = None
            n_probs = 0

        for i, county in enumerate(locs):
            Xin_i = X_all[start_ctx:end_idx+1, i, :].reshape(1, SEQ_LEN, 1, Din)
            pred_sc = model(torch.tensor(Xin_i, dtype=torch.float32, device=DEVICE))[0, 0].detach().cpu().numpy()  # (H,)

            if p_all is not None and i < n_probs:
                mask_vec = (p_all[:, i] >= float(cls_threshold)).astype(np.float32)
                pred_sc = pred_sc * mask_vec  # hard gating
            # else: no gating for this node if classifier didn't produce it

            y_mu = float(scalers["y_mu"].flatten()[0])
            y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0
            pred = (pred_sc * y_sd) + y_mu
            pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

            rows.append(pd.DataFrame({
                "timestamp": ts_future,
                "location": county,
                "pred": pred.astype(float)
            }))

    if rows:
        df_pred = pd.concat(rows, ignore_index=True)
    else:
        df_pred = pd.DataFrame(columns=["timestamp", "location", "pred"])
    return df_pred

# ---------------- training/infer (with optional gating in loss/infer) ----------------
def _masked_mse(pred, target, weight_or_mask):
    se = (pred - target) ** 2
    w = weight_or_mask
    num = (se * w).sum()
    denom = w.sum().clamp_min(1.0)
    return num / denom

def train_model(X, Y, din, horizon, n_nodes, eval_callback=None, classifier=None, gating_mode="soft", cls_threshold=0.5):
    if len(X) == 0:
        return None
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = STGNNForecaster(in_channels=din, horizon=horizon, num_nodes=n_nodes).to(DEVICE)

    enc_params = list(model.patch.parameters())
    enc_ids = set(id(p) for p in enc_params)
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]

    if FREEZE_ENCODER:
        opt = torch.optim.Adam(other_params, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        opt = torch.optim.Adam(
            [
                {"params": enc_params,   "lr": ENCODER_LR},
                {"params": other_params, "lr": LR}
            ],
            weight_decay=WEIGHT_DECAY
        )

    mse = nn.MSELoss()

    if classifier is not None:
        classifier.to(DEVICE)
        classifier.eval()
        if STOP_GRAD_GATING:
            for p in classifier.parameters():
                p.requires_grad = False

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)  # xb: (B,T,1,F), yb: (B,1,H)

            if TEMPORAL_DROP_PROB > 0.0:
                with torch.no_grad():
                    drop_mask = (torch.rand(xb.size(0), xb.size(1), 1, 1, device=xb.device) < TEMPORAL_DROP_PROB).float()
                    xb = xb * (1.0 - drop_mask)

            opt.zero_grad()
            pred = model(xb)  # (B,1,H)

            if classifier is None:
                loss = mse(pred, yb)
            else:
                with torch.no_grad():
                    logits = classifier(xb)  # (B,H,1) because N=1 in training
                    p = torch.sigmoid(logits)  # (B,H,1)
                    if gating_mode == "hard":
                        w = (p >= cls_threshold).float()
                    else:
                        w = p  # soft weighting
                loss = _masked_mse(pred, yb, w.permute(0,2,1))  # (B,1,H)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s")
        if eval_callback is not None:
            eval_callback(model, epoch + 1)

    return model

@torch.no_grad()
def infer(model, Xin):
    model.eval()
    out = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE))
    return out.cpu().numpy()

# ---------------- pipeline helpers (train windows only) ----------------
def prepare_training(ds_train, horizon):
    X, Y, Din, scalers, aux = prepare_training_split(ds_train, horizon, TRAIN_LAST_DAY)
    return X, Y, Din, scalers, aux

# ---------------- test-time prediction (per-county, with spatial gating SAFE) ----------------
@torch.no_grad()
def predict_per_county(ds_train, model, scalers, horizon, ts_future, classifier=None, cls_threshold=0.5):
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    T, N, F = w.shape
    Din = 1 + F

    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)

    if classifier is not None and T >= SEQ_LEN:
        Xin_all = X_all[-SEQ_LEN:, :, :].reshape(1, SEQ_LEN, N, Din)
        xb_all  = torch.tensor(Xin_all, dtype=torch.float32, device=DEVICE)
        logits_all = classifier(xb_all)[0].detach().cpu().numpy()  # (H,N_probs)
        p_all  = 1.0 / (1.0 + np.exp(-logits_all))
        n_probs = p_all.shape[1]
        n_locs  = len(locs)
        if n_probs != n_locs:
            print(f"[Classifier] Warning: N_probs ({n_probs}) != N_locs ({n_locs}); gating aligned by index where available.")
    else:
        p_all = None
        n_probs = 0

    rows = []
    for i, county in enumerate(locs):
        if T < SEQ_LEN:
            pred_sc = np.zeros(horizon, dtype=float)
        else:
            Xin_i = X_all[-SEQ_LEN:, i, :].reshape(1, SEQ_LEN, 1, Din)
            pred_sc = infer(model, Xin_i)[0, 0]
            if p_all is not None and i < n_probs:
                pred_sc = pred_sc * (p_all[:, i] >= float(cls_threshold)).astype(np.float32)

        pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)
        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))
    return pd.concat(rows, ignore_index=True)

# ---------------- main ----------------
def main():
    # Load train + test
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))

    _ = get_all_features(ds_train)

    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))

    zero_rmse_24 = _all_zero_rmse_for_truth(df_t24)
    zero_rmse_48 = _all_zero_rmse_for_truth(df_t48)

    # ==== 24h ====
    X24, Y24, Din, scalers24, aux24 = prepare_training(ds_train, horizon=H24)
    n_nodes = 1  # single-node per sample

    if USE_CLASSIFIER_GATING and os.path.exists(CLS_FROM_CKPT):
        classifier24 = CBOutageClassifier(
            in_channels=Din, horizon=H24, ckpt_path=CLS_FROM_CKPT,
            pooling=CLASSIFIER_POOL, spatial_mix=CLASSIFIER_SPATIAL_MIX
        ).to(DEVICE)
    else:
        classifier24 = None
        if USE_CLASSIFIER_GATING:
            print(f"[Classifier] Warning: ckpt not found at {CLS_FROM_CKPT}. Gating disabled.")

    df_truth_val24 = truth_wide_for_val(ds_train, aux24, H24, TRAIN_LAST_DAY)
    if classifier24 is not None and AUTO_TUNE_THRESHOLD:
        df_probs_val24 = predict_val_probs(ds_train, classifier24, aux24, H24, TRAIN_LAST_DAY)
        best_tau24, (z2nz24, nz2z24) = tune_threshold(df_truth_val24, df_probs_val24, ZERO2NONZERO_TARGET, THRESH_GRID)
        print_cls_metrics("Tuning 24h", df_truth_val24, df_probs_val24, best_tau24)
        tau24 = best_tau24
    else:
        df_probs_val24 = None
        tau24 = CLS_THRESHOLD

    val_cb24 = make_val_callback_rolling(
        ds_train, scalers24, H24, aux24, label="24h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts24, df_truth_test=df_t24, zero_rmse_test=zero_rmse_24,
        classifier=classifier24, cls_threshold=tau24,
        df_truth_val_cached=df_truth_val24, df_probs_val_cached=df_probs_val24
    )
    mdl24 = train_model(X24, Y24, Din, H24, n_nodes, eval_callback=val_cb24,
                        classifier=classifier24, gating_mode=GATING_MODE, cls_threshold=tau24)

    # ==== 48h ====
    X48, Y48, _, scalers48, aux48 = prepare_training(ds_train, horizon=H48)
    if USE_CLASSIFIER_GATING and os.path.exists(CLS_FROM_CKPT):
        classifier48 = CBOutageClassifier(
            in_channels=Din, horizon=H48, ckpt_path=CLS_FROM_CKPT,
            pooling=CLASSIFIER_POOL, spatial_mix=CLASSIFIER_SPATIAL_MIX
        ).to(DEVICE)
    else:
        classifier48 = None

    df_truth_val48 = truth_wide_for_val(ds_train, aux48, H48, TRAIN_LAST_DAY)
    if classifier48 is not None and AUTO_TUNE_THRESHOLD:
        df_probs_val48 = predict_val_probs(ds_train, classifier48, aux48, H48, TRAIN_LAST_DAY)
        best_tau48, (z2nz48, nz2z48) = tune_threshold(df_truth_val48, df_probs_val48, ZERO2NONZERO_TARGET, THRESH_GRID)
        print_cls_metrics("Tuning 48h", df_truth_val48, df_probs_val48, best_tau48)
        tau48 = best_tau48
    else:
        df_probs_val48 = None
        tau48 = CLS_THRESHOLD

    val_cb48 = make_val_callback_rolling(
        ds_train, scalers48, H48, aux48, label="48h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts48, df_truth_test=df_t48, zero_rmse_test=zero_rmse_48,
        classifier=classifier48, cls_threshold=tau48,
        df_truth_val_cached=df_truth_val48, df_probs_val_cached=df_probs_val48
    )
    mdl48 = train_model(X48, Y48, Din, H48, n_nodes, eval_callback=val_cb48,
                        classifier=classifier48, gating_mode=GATING_MODE, cls_threshold=tau48)

    # ----- Test evaluation using best models, then write CSVs -----
    if mdl24 is not None:
        best24_path = os.path.join(OUT_DIR, "best_enc-dec_24h.pt")
        if os.path.exists(best24_path):
            ckpt24 = torch.load(best24_path, map_location=DEVICE)
            try:
                mdl24.load_state_dict(ckpt24["model_state_dict"])
                rmse24_val = ckpt24.get("rmse_select", ckpt24.get("rmse", None))
                ep24 = ckpt24.get("epoch", None)
                if isinstance(rmse24_val, (float, int)):
                    print(f"[24h] Loaded best checkpoint (epoch {ep24}, Val selected RMSE={rmse24_val:.4f}, best_by={ckpt24.get('best_by','overall')})")
                else:
                    print(f"[24h] Loaded best checkpoint (epoch {ep24})")
            except Exception as e:
                print(f"[24h] Warning: failed to load best checkpoint: {e}")
        df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24,
                                  classifier=classifier24, cls_threshold=tau24)
        test_rmse_24 = per_county_rmse_mean(df_t24, df24)
        print(f"[Test 24h] Testing RMSE={test_rmse_24:.4f} (all-zero RMSE={zero_rmse_24:.4f})")
    else:
        df24 = None

    if mdl48 is not None:
        best48_path = os.path.join(OUT_DIR, "best_enc-dec_48h.pt")
        if os.path.exists(best48_path):
            ckpt48 = torch.load(best48_path, map_location=DEVICE)
            try:
                mdl48.load_state_dict(ckpt48["model_state_dict"])
                rmse48_val = ckpt48.get("rmse_select", ckpt48.get("rmse", None))
                ep48 = ckpt48.get("epoch", None)
                if isinstance(rmse48_val, (float, int)):
                    print(f"[48h] Loaded best checkpoint (epoch {ep48}, Val selected RMSE={rmse48_val:.4f}, best_by={ckpt48.get('best_by','overall')})")
                else:
                    print(f"[48h] Loaded best checkpoint (epoch {ep48})")
            except Exception as e:
                print(f"[48h] Warning: failed to load best checkpoint: {e}")
        df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48,
                                  classifier=classifier48, cls_threshold=tau48)
        test_rmse_48 = per_county_rmse_mean(df_t48, df48)
        print(f"[Test 48h] Testing RMSE={test_rmse_48:.4f} (all-zero RMSE={zero_rmse_48:.4f})")
    else:
        df48 = None

    if df24 is None:
        locs = list(map(str, ds_train.location.values))
        df24 = pd.DataFrame({"timestamp": np.repeat(ts24, len(locs)),
                             "location": np.tile(locs, len(ts24)),
                             "pred": 0.0})
    if df48 is None:
        locs = list(map(str, ds_train.location.values))
        df48 = pd.DataFrame({"timestamp": np.repeat(ts48, len(locs)),
                             "location": np.tile(locs, len(ts48)),
                             "pred": 0.0})

    out24 = os.path.join(OUT_DIR, "stemgnn_pred_24h.csv")
    out48 = os.path.join(OUT_DIR, "stemgnn_pred_48h.csv")
    df24.to_csv(out24, index=False)
    df48.to_csv(out48, index=False)

    print("Saved predictions:")
    print(f"  {out24}")
    print(f"  {out48}")

    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
