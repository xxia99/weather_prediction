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
import matplotlib.pyplot as plt
from demo_evaluate import load_truth_wide, per_county_rmse_mean
# (we no longer call event_stratified_report to avoid duplicate prints)

# ---------------- config ----------------
SPLIT_DIR = "data/"
# OUT_DIR   = "results_seq_mse_pre-CB_use-moe/update_restore_best"
OUT_DIR   = "results_seq_mse_pre-CB_surge/update_new_lora_ABCs"  # <<< NEW: simple name bump
# OUT_DIR   = "results_seq_mse_pre-patch_use-moe"
# OUT_DIR   = "results_seq_mse_non-pre-patch_use-moe/2experts"
os.makedirs(OUT_DIR, exist_ok=True)

# core training hyperparameters
SEQ_LEN       = 24
BATCH_SIZE    = 64
EPOCHS        = 40
LR            = 5e-4
WEIGHT_DECAY  = 5e-5
DEVICE        = "cuda:1" if torch.cuda.is_available() else "cpu"

# === temporal dropout probability (drop whole timesteps during training) ===
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

# === Pretrained encoder options ===
if "non" in OUT_DIR:
    USE_PRETRAINED_ENCODER = False
else:
    PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_validation.pt"
ENCODER_LR              = 3e-4
FREEZE_ENCODER          = False

# === Use Chronos-Bolt PEFT encoder instead of PatchTST ===
if "patch" in OUT_DIR:
    USE_CB_ENCODER = False
else:
    USE_CB_ENCODER = True
# CB_ENCODER_CKPT         = "results_lora/new/pretrained_cb_encoder_peft_independent_H24.pt"
CB_ENCODER_CKPT = "results_lora/new/pretrained_cb_encoder_peft_independent_H24.pt"

# === Day split for in-train validation ===
TRAIN_LAST_DAY = 80

# === Event-stratified validation settings ===
EVENT_METHOD   = "mean"
STORM_TOP_Q    = 0.80
CALM_BOTTOM_Q  = 0.50
CALM_N         = 10
BEST_BY        = "overall"

# ========= zero/nonzero diagnostics =========
ZERO_EPS = 1e-8

# ========= MoE settings =========
USE_MOE        = True
MOE_NUM_EXPERTS= 4
MOE_TOP_K      = 2
MOE_HIDDEN     = 128
MOE_DROPOUT    = 0.10
W_MOE_AUX      = 1e-2

# ========= Probabilistic head with quantiles =========
USE_QUANTILES   = True
QUANTILES       = [0.1, 0.5, 0.9]
PRED_QUANTILE   = 0.90   # will be overridden adaptively by surge prob (A)

# ---------------- utils ----------------
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
            d_model=d_model, nhead=nhead, dim_feedforward=ff, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, F = x.shape
        P, S = self.patch_len, self.stride
        assert F == self.in_channels, f"PatchTST got F={F}, expected {self.in_channels}"

        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)

        L = (T + S - 1) // S
        T_pad = (L - 1) * S + P
        if T_pad < T:
            L += 1
            T_pad = (L - 1) * S + P
        if T_pad > T:
            pad = torch.zeros(B * N, T_pad - T, F, device=x.device, dtype=x.dtype)
            x_bn = torch.cat([x_bn, pad], dim=1)

        x_unf = x_bn.unfold(dimension=1, size=P, step=S)          # (B*N, L, P, F)
        x_unf = x_unf.contiguous().view(B * N, L, P * F)

        tokens = self.proj(x_unf)
        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)

        tokens_rep = tokens.repeat_interleave(P, dim=1)
        steps = tokens_rep[:, :T, :]

        h = steps.view(B, N, T, self.d_model).permute(0, 2, 1, 3)
        return h

# -------------------- Chronos-Bolt PEFT encoder --------------------
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
        self.load_state_dict(ck["state_dict"], strict=True)
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

# -------------------- StemGNN primitives --------------------
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

# -------------------- MoE readout (unchanged API) --------------------
class MoEReadout(nn.Module):
    """
    Inputs:  h_last (B, M, D)  # NOTE: M can be N or N*H (we'll use N*H)
    Outputs: y (B, M, OUT_DIM), aux_loss, probs_mean
    """
    def __init__(self, d_in, out_dim, num_experts=4, hidden=128, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(max(1, min(top_k, self.num_experts)))
        self.router = nn.Linear(d_in, self.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim)
            ) for _ in range(self.num_experts)
        ])

    def forward(self, h_last):
        B, M, D = h_last.shape
        logits = self.router(h_last)
        probs  = torch.softmax(logits, dim=-1)

        if self.top_k < self.num_experts:
            topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
            norm = torch.clamp(topv.sum(dim=-1, keepdim=True), min=1e-9)
            topw = topv / norm
        else:
            topi = torch.arange(self.num_experts, device=h_last.device).view(1,1,self.num_experts).expand(B,M,self.num_experts)
            topw = probs

        expert_outs = torch.stack([exp(h_last) for exp in self.experts], dim=-1)   # (B,M,OUT,E)
        gather_idx = topi.unsqueeze(-2).expand(B, M, expert_outs.size(-2), topw.size(-1))
        selected = expert_outs.gather(dim=-1, index=gather_idx)                    # (B,M,OUT,K)
        y = (selected * topw.unsqueeze(-2)).sum(dim=-1)                            # (B,M,OUT)

        probs_mean = probs.mean(dim=(0,1))
        eps = 1e-9
        aux = (probs_mean * (torch.log(probs_mean + eps) + math.log(self.num_experts))).sum()
        return y, aux, probs_mean

# -------------------- Horizon-aware decoder (B) --------------------
class HorizonDecoder(nn.Module):
    """
    Produces per-horizon embeddings from context:
      inputs: last (B,N,D), hist_mean (B,N,D)
      returns: dec_feats (B,N,H,D)
    """
    def __init__(self, d_in, horizon):
        super().__init__()
        self.horizon = horizon
        self.query = nn.Parameter(torch.randn(horizon, d_in))  # H x D
        self.fc1 = nn.Linear(d_in * 2, d_in)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_in, d_in)  # keep D

    def forward(self, last, hist_mean):
        B, N, D = last.shape
        q = self.query.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # (B,N,H,D)
        ctx = torch.stack([last, hist_mean], dim=2).mean(2)            # (B,N,D)
        ctx = ctx.unsqueeze(2).expand(B, N, self.horizon, D)           # (B,N,H,D)
        x = torch.cat([q, ctx], dim=-1)                                # (B,N,H,2D)
        h = self.act(self.fc1(x))
        h = self.fc2(h)                                                # (B,N,H,D)
        return h

# -------------------- Forecaster with Surge Head (A) --------------------
class STGNNForecaster(nn.Module):
    """
    Encoder -> stacked Stem blocks -> horizon-aware decoder -> (MoE per-horizon) -> quantiles
    Also predicts surge probability over next H hours.
    """
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels
        self.n_quantiles = len(QUANTILES) if USE_QUANTILES else 1
        out_dim_per_h = self.n_quantiles

        # --- encoder ---
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
            if 'PRETRAINED_ENCODER_PATH' in globals() and PRETRAINED_ENCODER_PATH is not None and os.path.exists(PRETRAINED_ENCODER_PATH):
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

        # --- surge head (A) ---
        self.surge_head = nn.Sequential(
            nn.Linear(STEM_D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # --- horizon-aware decoder (B) ---
        self.hdec = HorizonDecoder(STEM_D_MODEL, self.horizon)

        # --- per-horizon MoE readout to quantiles ---
        if USE_MOE:
            self.readout_moe = MoEReadout(
                d_in=STEM_D_MODEL, out_dim=out_dim_per_h,
                num_experts=MOE_NUM_EXPERTS, hidden=MOE_HIDDEN,
                top_k=MOE_TOP_K, dropout=MOE_DROPOUT
            )
            self.readout = None
            print(f"[MoE] Enabled: E={MOE_NUM_EXPERTS}, top_k={MOE_TOP_K}, hidden={MOE_HIDDEN}, out_per_h={out_dim_per_h}")
        else:
            self.readout = nn.Linear(STEM_D_MODEL, out_dim_per_h)
            self.readout_moe = None

        self._last_aux_loss = None
        self._last_gate_mean = None

    def _encode_context(self, x):
        """Returns per-timestep features h (B,T,N,D)."""
        h = self.patch(x)
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(self, x):
        """
        Training: returns (yq, surge_p)
        Eval:     returns yq
        Shapes:
          yq: (B,N,H,Q) if USE_QUANTILES else (B,N,H)
          surge_p: (B,N)
        """
        B, T, N, F = x.shape
        h = self._encode_context(x)
        last = h[:, -1]                    # (B,N,D)
        hist_mean = h.mean(dim=1)          # (B,N,D)

        # (A) surge probability
        surge_p = self.surge_head(last).squeeze(-1)  # (B,N)

        # (B) horizon-aware embeddings
        dec_feats = self.hdec(last, hist_mean)       # (B,N,H,D)

        # readout to quantiles
        if self.readout_moe is not None:
            B_, N_, H_, D_ = dec_feats.shape
            flat = dec_feats.reshape(B_, N_ * H_, D_)                # (B, N*H, D)
            y_flat, aux, probs_mean = self.readout_moe(flat)         # (B, N*H, Q)
            self._last_aux_loss = aux
            self._last_gate_mean = probs_mean.detach()
            y = y_flat.view(B_, N_, H_, -1)                          # (B,N,H,Q)
        else:
            y = self.readout(dec_feats)                              # (B,N,H,Q or 1)
        if self.training:
            return y, surge_p
        else:
            return y

    # convenience getters
    def moe_aux_loss(self):
        return self._last_aux_loss

    def moe_gate_usage(self):
        return self._last_gate_mean

    # helper to fetch surge prob at inference (A)
    @torch.no_grad()
    def get_surge_prob(self, x):
        h = self._encode_context(x)
        last = h[:, -1]
        return self.surge_head(last).squeeze(-1)  # (B,N)

# ---------------- helpers for day split & windows ----------------
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
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)

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
    X_tr = np.concatenate(X_list, axis=0)
    Y_tr = np.concatenate(Y_list, axis=0)

    aux = {
        "X_all_scaled": X_all,
        "y_scaled": y_sc,
        "timestamps": ts,
        "days": days,
        "day_end_idx": day_end_idx
    }
    return X_tr, Y_tr, Din, scalers, aux

# ---------------- validation roll ----------------
@torch.no_grad()
def predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day=80):
    model.eval()
    X_all = aux["X_all_scaled"]
    ts    = aux["timestamps"]
    days  = aux["days"]
    day_end_idx = aux["day_end_idx"]
    N = X_all.shape[1]
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

        for i, county in enumerate(locs):
            Xin_i = X_all[start_ctx:end_idx+1, i, :].reshape(1, SEQ_LEN, 1, X_all.shape[2])
            pred_sc = infer(model, Xin_i)[0, 0]  # (H,)

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

# ========= zero/nonzero diagnostic ratios =========
def _zero_nonzero_ratios(df_truth_wide: pd.DataFrame, df_pred_long: pd.DataFrame, eps: float = ZERO_EPS):
    if df_truth_wide is None or df_truth_wide.empty or df_pred_long is None or df_pred_long.empty:
        return float("nan"), float("nan")

    df_t = df_truth_wide.copy()
    df_t["timestamp"] = pd.to_datetime(df_t["timestamp"], errors="coerce").astype("datetime64[ns]")
    df_t = df_t.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    tw = df_t.set_index("timestamp").sort_index()

    df_p = df_pred_long.copy()
    df_p["timestamp"] = pd.to_datetime(df_p["timestamp"], errors="coerce").astype("datetime64[ns]")
    df_p["location"]  = df_p["location"].astype(str)
    df_p = df_p.dropna(subset=["timestamp", "location"])
    df_p = df_p.groupby(["timestamp", "location"], as_index=False, sort=False)["pred"].mean()

    pw = df_p.pivot_table(index="timestamp", columns="location", values="pred", aggfunc="mean")
    pw = pw.sort_index()

    common_ts = tw.index.intersection(pw.index)
    if len(common_ts) == 0:
        return float("nan"), float("nan")
    tw = tw.loc[common_ts]
    pw = pw.loc[common_ts]

    loc_truth = list(tw.columns)
    loc_pred  = list(pw.columns)
    locs = sorted(set(loc_truth).intersection(loc_pred))
    if len(locs) == 0:
        return float("nan"), float("nan")

    y_true = tw[locs].to_numpy()
    y_pred = pw[locs].to_numpy()

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan"), float("nan")

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    true_zero = (np.abs(y_true) <= eps)
    pred_zero = (np.abs(y_pred) <= eps)

    denom_zero = np.sum(true_zero)
    z_to_nz = np.sum(true_zero & (~pred_zero)) / denom_zero if denom_zero > 0 else float("nan")

    denom_nz = np.sum(~true_zero)
    nz_to_z = np.sum((~true_zero) & pred_zero) / denom_nz if denom_nz > 0 else float("nan")

    return float(z_to_nz), float(nz_to_z)

# ---- helpers for event selection (calm/storm) ----
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

def make_val_callback_rolling(ds_train, scalers, horizon, aux, label,
                              train_last_day=80,
                              ts_future_test=None, df_truth_test=None, zero_rmse_test=float("nan")):
    best = {"rmse": float("inf"), "epoch": -1}
    df_truth_val = truth_wide_for_val(ds_train, aux, horizon, train_last_day)

    def _callback(model, epoch):
        df_pred_val = predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day)

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

        zpnz_model, nzpz_model = _zero_nonzero_ratios(df_truth_val, df_pred_val, eps=ZERO_EPS)
        if not df_truth_val.empty:
            locs = [c for c in df_truth_val.columns if c != "timestamp"]
            df_zero_long = pd.DataFrame({
                "timestamp": np.repeat(df_truth_val["timestamp"].values, len(locs)),
                "location": np.tile(locs, len(df_truth_val)),
                "pred": 0.0
            })
            zpnz_zero, nzpz_zero = _zero_nonzero_ratios(df_truth_val, df_zero_long, eps=ZERO_EPS)
        else:
            zpnz_zero, nzpz_zero = float("nan"), float("nan")

        if np.isfinite(rmse_overall):
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE={rmse_overall:.4f} (all-zero={zero_overall:.4f})")
        else:
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE=nan (all-zero=nan)")

        print(f"[Val {label}] Epoch {epoch}: Zero→Nonzero ratio={zpnz_model if np.isfinite(zpnz_model) else float('nan'):.4f} "
              f"(all-zero={zpnz_zero if np.isfinite(zpnz_zero) else float('nan'):.4f}) | "
              f"Nonzero→Zero ratio={nzpz_model if np.isfinite(nzpz_model) else float('nan'):.4f} "
              f"(all-zero={nzpz_zero if np.isfinite(nzpz_zero) else float('nan'):.4f})")

        if np.isfinite(rmse_calm):
            print(f"[EventEval {label}] Calm-heavy ({n_calm} days): RMSE={rmse_calm:.4f} (all-zero={zero_calm:.4f})")
        else:
            print(f"[EventEval {label}] Calm-heavy: RMSE=nan (all-zero=nan)")

        if np.isfinite(rmse_storm):
            print(f"[EventEval {label}] Storm-heavy ({n_storm} days): RMSE={rmse_storm:.4f} (all-zero={zero_storm:.4f})")
        else:
            print(f"[EventEval {label}] Storm-heavy: RMSE=nan (all-zero=nan)")

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

            ts_map = pd.DataFrame({"timestamp": pd.to_datetime(aux["timestamps"]), "day": aux["days"]})
            df_pred_day = df_pred_val.merge(ts_map, on="timestamp", how="left")
            df_truth_day = df_truth_val.merge(ts_map, on="timestamp", how="left")

            unique_days = sorted(df_pred_day["day"].dropna().unique())
            for d in unique_days:
                pred_d = df_pred_day[df_pred_day["day"] == d].drop(columns="day")
                truth_d = df_truth_day[df_truth_day["day"] == d].drop(columns="day")
                pred_path = os.path.join(OUT_DIR, f"val_pred_epoch{epoch}_day{int(d)}_{label}.csv")
                truth_path = os.path.join(OUT_DIR, f"val_truth_epoch{epoch}_day{int(d)}_{label}.csv")
                pred_d.to_csv(pred_path, index=False)
                truth_d.to_csv(truth_path, index=False)
                print(f"[Val {label}] Saved day {int(d)} prediction/truth data: {pred_path}, {truth_path}")

            if unique_days:
                fig, axes = plt.subplots(len(unique_days), 1, figsize=(8, 3 * len(unique_days)), sharex=False)
                if len(unique_days) == 1:
                    axes = [axes]
                for ax, d in zip(axes, unique_days):
                    truth_d = df_truth_day[df_truth_day["day"] == d].set_index("timestamp").drop(columns="day")
                    truth_mean = truth_d.mean(axis=1).reset_index(name="truth")
                    pred_mean = (
                        df_pred_day[df_pred_day["day"] == d]
                        .groupby("timestamp")["pred"].mean().reset_index()
                    )
                    merged = pd.merge(truth_mean, pred_mean, on="timestamp", how="inner").sort_values("timestamp")
                    ax.plot(merged["timestamp"], merged["truth"], label="Truth")
                    ax.plot(merged["timestamp"], merged["pred"], label="Prediction")
                    ax.set_title(f"Day {int(d)}")
                    ax.set_xlabel("timestamp")
                    ax.set_ylabel("outage")
                    ax.legend()
                fig_path = os.path.join(OUT_DIR, f"val_curve_epoch{epoch}_{label}.png")
                fig.tight_layout()
                fig.savefig(fig_path)
                plt.close(fig)
                print(f"[Val {label}] Saved prediction vs truth plot: {fig_path}")

        if ts_future_test is not None and df_truth_test is not None and len(ts_future_test) > 0:
            df_pred_test = predict_per_county(ds_train, model, scalers, horizon, ts_future_test)
            rmse_test = per_county_rmse_mean(df_truth_test, df_pred_test)

            zpnz_t, nzpz_t = _zero_nonzero_ratios(df_truth_test, df_pred_test, eps=ZERO_EPS)
            locs_t = [c for c in df_truth_test.columns if c != "timestamp"]
            df_zero_long_t = pd.DataFrame({
                "timestamp": np.repeat(df_truth_test["timestamp"].values, len(locs_t)),
                "location": np.tile(locs_t, len(df_truth_test)),
                "pred": 0.0
            })
            zpnz_t0, nzpz_t0 = _zero_nonzero_ratios(df_truth_test, df_zero_long_t, eps=ZERO_EPS)

            print(f"[Val {label}] Epoch {epoch}: Testing RMSE={rmse_test:.4f} (all-zero RMSE={zero_rmse_test:.4f})")
            print(f"[Val {label}] Epoch {epoch}: Test Zero→Nonzero ratio={zpnz_t if np.isfinite(zpnz_t) else float('nan'):.4f} "
                  f"(all-zero={zpnz_t0 if np.isfinite(zpnz_t0) else float('nan'):.4f}) | "
                  f"Nonzero→Zero ratio={nzpz_t if np.isfinite(nzpz_t) else float('nan'):.4f} "
                  f"(all-zero={nzpz_t0 if np.isfinite(nzpz_t0) else float('nan'):.4f})")

    return _callback

# ---------------- pinball loss (C adds a trend term later) ----------------
def _pinball_loss(yq, y_true, q_list):
    qs = torch.tensor(q_list, device=yq.device, dtype=yq.dtype).view(1,1,1,-1)
    diff = y_true.unsqueeze(-1) - yq
    loss = torch.maximum(qs * diff, (qs - 1.0) * diff)
    return loss.mean()

# ---------------- training/infer ----------------
def _delta(x, dim=-1):
    return x[..., 1:] - x[..., :-1]

def train_model(X, Y, din, horizon, n_nodes, eval_callback=None):
    """
    X: (num_samples, SEQ_LEN, N, din)
    Y: (num_samples, N, horizon)   # scaled
    """
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

    loss_fn_point = nn.MSELoss()
    bce = nn.BCELoss()

    # --- compute a fixed surge threshold on training targets (scaled) ---
    thr_cached = float(np.quantile(Y.reshape(-1), 0.90))
    train_model._thr = thr_cached  # store for reference

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)  # xb: (B, T, N, F), yb: (B, N, H)

            if TEMPORAL_DROP_PROB > 0.0:
                with torch.no_grad():
                    drop_mask = (torch.rand(xb.size(0), xb.size(1), 1, 1, device=xb.device) < TEMPORAL_DROP_PROB).float()
                    xb = xb * (1.0 - drop_mask)

            opt.zero_grad()
            # forward returns (yq, surge_p) in training mode
            y_out, surge_p = model(xb)

            # -- main loss --
            if USE_QUANTILES:
                loss_main = _pinball_loss(y_out, yb, QUANTILES)    # (C) base term
                # for trend term use median quantile for stability
                qs = np.array(QUANTILES, dtype=float)
                mid = int(np.argmin(np.abs(qs - 0.5)))
                pred_mid = y_out[..., mid]                         # (B,N,H)
                d_pred = _delta(pred_mid, dim=-1)
            else:
                loss_main = loss_fn_point(y_out, yb)
                d_pred = _delta(y_out, dim=-1)

            # -- (A) surge BCE supervision --
            with torch.no_grad():
                surge_label = (yb.max(dim=-1).values > train_model._thr).float()  # (B,N)
            loss_surge = bce(surge_p, surge_label)

            # -- (C) trend/spike loss (heavier weight on up-jumps) --
            d_true = _delta(yb, dim=-1)
            jump_w = (d_true.relu())
            jump_w = jump_w / (jump_w.mean() + 1e-6)
            loss_trend = ((d_pred - d_true).pow(2) * (1.0 + 2.0 * jump_w)).mean()

            # -- MoE aux loss if present --
            aux = model.moe_aux_loss()
            loss = loss_main + 0.2 * loss_surge + 0.1 * loss_trend + (W_MOE_AUX * aux if aux is not None else 0.0)

            # optional: oversample/reweight batches with many surges
            batch_weight = 1.0 + surge_label.mean().item()  # up to ~2x
            loss = loss * batch_weight

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        elapsed = time.time() - start_time
        lr_str = ", ".join(f"{pg['lr']:.2e}" for pg in opt.param_groups)

        gate_mean = model.moe_gate_usage()
        if gate_mean is not None:
            gm = gate_mean.detach().cpu().numpy()
            gm_str = " | ".join([f"E{i}:{gm[i]:.2f}" for i in range(len(gm))])
            print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s | LR: {lr_str} | MoE usage={gm_str}")
        else:
            print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s | LR: {lr_str}")

        if eval_callback is not None:
            eval_callback(model, epoch + 1)

    return model

@torch.no_grad()
def infer(model, Xin):
    """
    Xin: (1, SEQ_LEN, N, F)
    Returns: (1, N, H) point forecast with **adaptive quantile** based on surge prob (A).
    """
    model.eval()
    Xin_t = torch.tensor(Xin, dtype=torch.float32).to(DEVICE)
    out = model(Xin_t)  # yq or y
    if USE_QUANTILES:
        # compute surge prob from helper
        surge_p = model.get_surge_prob(Xin_t)[0, 0].item()
        use_q = 0.90 if surge_p >= 0.5 else 0.50
        qs = np.array(QUANTILES, dtype=float)
        idx = int(np.argmin(np.abs(qs - use_q)))
        out = out[..., idx]  # (1, N, H)
    return out.cpu().numpy()

# ---------------- pipeline helpers (train windows only) ----------------
def prepare_training(ds_train, horizon):
    X, Y, Din, scalers, aux = prepare_training_split(ds_train, horizon, TRAIN_LAST_DAY)
    return X, Y, Din, scalers, aux

# ---------------- test-time prediction (per county) ----------------
def predict_per_county(ds_train, model, scalers, horizon, ts_future):
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    T, N, F = w.shape
    Din = 1 + F

    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)

    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)

    rows = []
    for i, county in enumerate(locs):
        if T < SEQ_LEN:
            pred_sc = np.zeros(horizon, dtype=float)
        else:
            Xin_i = X_all[-SEQ_LEN:, i, :].reshape(1, SEQ_LEN, 1, Din)
            pred_sc = infer(model, Xin_i)[0, 0]

        pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))
    return pd.concat(rows, ignore_index=True)

def make_eval_callback(ds_train, scalers, horizon, ts_future, df_truth, label):
    best = {"rmse": float("inf"), "epoch": -1}

    def _callback(model, epoch):
        df_pred = predict_per_county(ds_train, model, scalers, horizon, ts_future)
        rmse = per_county_rmse_mean(df_truth, df_pred)
        print(f"[Eval {label}] Epoch {epoch}: RMSE={rmse:.4f}")

        if np.isfinite(rmse) and rmse < best["rmse"]:
            best["rmse"] = rmse
            best["epoch"] = epoch
            save_path = os.path.join(OUT_DIR, f"best_enc-dec_{label}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "label": label,
                    "rmse": rmse,
                    "model_state_dict": model.state_dict(),
                },
                save_path,
            )
            print(f"[Eval {label}] New best RMSE {rmse:.4f} at epoch {epoch}. Saved: {save_path}")

            pred_path = os.path.join(OUT_DIR, f"val_pred_epoch{epoch}_{label}.csv")
            truth_path = os.path.join(OUT_DIR, f"val_truth_epoch{epoch}_{label}.csv")
            df_pred.to_csv(pred_path, index=False)
            df_truth.to_csv(truth_path, index=False)

            truth_mean = df_truth.set_index("timestamp").mean(axis=1).reset_index(name="truth")
            pred_mean = df_pred.groupby("timestamp")["pred"].mean().reset_index()
            merged = pd.merge(truth_mean, pred_mean, on="timestamp", how="inner").sort_values("timestamp")
            plt.figure()
            plt.plot(merged["timestamp"], merged["truth"], label="Truth")
            plt.plot(merged["timestamp"], merged["pred"], label="Prediction")
            plt.legend()
            plt.xlabel("timestamp")
            plt.ylabel("outage")
            plt.title(f"Validation Prediction vs Truth ({label}) epoch {epoch}")
            fig_path = os.path.join(OUT_DIR, f"val_curve_epoch{epoch}_{label}.png")
            plt.savefig(fig_path)
            plt.close()
            print(f"[Eval {label}] Saved prediction/truth data: {pred_path}, {truth_path}")
            print(f"[Eval {label}] Saved prediction vs truth plot: {fig_path}")

    return _callback

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
    n_nodes = 1
    val_cb24 = make_val_callback_rolling(
        ds_train, scalers24, H24, aux24, label="24h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts24, df_truth_test=df_t24, zero_rmse_test=zero_rmse_24
    )
    mdl24 = train_model(X24, Y24, Din, H24, n_nodes, eval_callback=val_cb24)

    # ==== 48h ====
    X48, Y48, _, scalers48, aux48 = prepare_training(ds_train, horizon=H48)
    val_cb48 = make_val_callback_rolling(
        ds_train, scalers48, H48, aux48, label="48h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts48, df_truth_test=df_t48, zero_rmse_test=zero_rmse_48
    )
    mdl48 = train_model(X48, Y48, Din, H48, n_nodes, eval_callback=val_cb48)

    # ----- Test evaluation using best models, then write CSVs -----
    # 24h
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
        df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24)
        test_rmse_24 = per_county_rmse_mean(df_t24, df24)
        print(f"[Test 24h] Testing RMSE={test_rmse_24:.4f} (all-zero RMSE={zero_rmse_24:.4f})")

        zpnz_24, nzpz_24 = _zero_nonzero_ratios(df_t24, df24, eps=ZERO_EPS)
        locs24 = [c for c in df_t24.columns if c != "timestamp"]
        df_zero_long_24 = pd.DataFrame({
            "timestamp": np.repeat(df_t24["timestamp"].values, len(locs24)),
            "location": np.tile(locs24, len(df_t24)),
            "pred": 0.0
        })
        zpnz_24_0, nzpz_24_0 = _zero_nonzero_ratios(df_t24, df_zero_long_24, eps=ZERO_EPS)
        print(f"[Test 24h] Zero→Nonzero ratio={zpnz_24 if np.isfinite(zpnz_24) else float('nan'):.4f} "
              f"(all-zero={zpnz_24_0 if np.isfinite(zpnz_24_0) else float('nan'):.4f}) | "
              f"Nonzero→Zero ratio={nzpz_24 if np.isfinite(nzpz_24) else float('nan'):.4f} "
              f"(all-zero={nzpz_24_0 if np.isfinite(nzpz_24_0) else float('nan'):.4f})")
    else:
        df24 = None

    # 48h
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
        df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48)
        test_rmse_48 = per_county_rmse_mean(df_t48, df48)
        print(f"[Test 48h] Testing RMSE={test_rmse_48:.4f} (all-zero RMSE={zero_rmse_48:.4f})")

        zpnz_48, nzpz_48 = _zero_nonzero_ratios(df_t48, df48, eps=ZERO_EPS)
        locs48 = [c for c in df_t48.columns if c != "timestamp"]
        df_zero_long_48 = pd.DataFrame({
            "timestamp": np.repeat(df_t48["timestamp"].values, len(locs48)),
            "location": np.tile(locs48, len(df_t48)),
            "pred": 0.0
        })
        zpnz_48_0, nzpz_48_0 = _zero_nonzero_ratios(df_t48, df_zero_long_48, eps=ZERO_EPS)
        print(f"[Test 48h] Zero→Nonzero ratio={zpnz_48 if np.isfinite(zpnz_48) else float('nan'):.4f} "
              f"(all-zero={zpnz_48_0 if np.isfinite(zpnz_48_0) else float('nan'):.4f}) | "
              f"Nonzero→Zero ratio={nzpz_48 if np.isfinite(nzpz_48) else float('nan'):.4f} "
              f"(all-zero={nzpz_48_0 if np.isfinite(nzpz_48_0) else float('nan'):.4f})")
    else:
        df48 = None

    # Fallbacks
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
