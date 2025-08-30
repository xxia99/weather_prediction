import os
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import xarray as xr
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# NEW: schedulers
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import matplotlib.pyplot as plt
from demo_evaluate import load_truth_wide, per_county_rmse_mean

# ======================== config ========================
SPLIT_DIR = "data/"
OUT_DIR   = "results_seq_mse_pre-CB_use-moe/two_stage_cls_blend_warmcos_ema"
os.makedirs(OUT_DIR, exist_ok=True)

SURGE_CSV_PATH       = "results_surge/surge_ranking_mean.csv"
USE_SURGE_RANK_CSV   = True

# ---- core training ----
SEQ_LEN       = 24
BATCH_SIZE    = 64
EPOCHS        = 40
LR            = 5e-4
WEIGHT_DECAY  = 5e-5
DEVICE        = "cuda:1" if torch.cuda.is_available() else "cpu"

# NEW: tiny temporal mask
TEMPORAL_DROP_PROB = 0.03

# ---- PatchTST encoder ----
PATCH_LEN      = 2
PATCH_STRIDE   = None
PATCH_D_MODEL  = 64
PATCH_NHEAD    = 4
PATCH_NLAYERS  = 2
PATCH_DROPOUT  = 0.05
PATCH_FF       = 4 * PATCH_D_MODEL

# ---- StemGNN ----
STEM_D_MODEL  = 64
STEM_K_ORDER  = 2
STEM_NBLOCKS  = 2
STEM_D_EMB    = 32
STEM_DROPOUT  = 0.03

# ---- pretrained encoder options ----
if "non" in OUT_DIR:
    USE_PRETRAINED_ENCODER = False
else:
    PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_validation.pt"
ENCODER_LR              = 3e-4
FREEZE_ENCODER          = False

# ---- Chronos-Bolt PEFT encoder or PatchTST ----
if "patch" in OUT_DIR:
    USE_CB_ENCODER = False
else:
    USE_CB_ENCODER = True
CB_ENCODER_CKPT = "results_lora/new/pretrained_cb_encoder_peft_independent_H24.pt"

# ---- day split ----
TRAIN_LAST_DAY = 80

# ---- event-stratified validation ----
EVENT_METHOD   = "mean"
STORM_TOP_Q    = 0.80
CALM_BOTTOM_Q  = 0.50
CALM_N         = 10
BEST_BY        = "overall"

# ---- zero/nonzero diagnostics ----
ZERO_EPS = 1e-8

# ---- MoE ----
USE_MOE        = True
MOE_NUM_EXPERTS= 4
MOE_TOP_K      = 2
MOE_HIDDEN     = 128
MOE_DROPOUT    = 0.10
W_MOE_AUX      = 1e-2

# ---- Quantiles ----
USE_QUANTILES   = True
QUANTILES       = [0.1, 0.5, 0.9]
PRED_QUANTILE   = 0.90

# ======================== NEW: two-stage surge classifier ========================
REGIME_NAMES = ["calm", "moderate", "storm"]
REGIME_QWEIGHTS = {
    "calm":     np.array([0.50, 0.45, 0.05], dtype=np.float32),
    "moderate": np.array([0.20, 0.50, 0.30], dtype=np.float32),
    "storm":    np.array([0.10, 0.30, 0.60], dtype=np.float32),
}

# classifier training
CLS_EPOCHS      = 20
CLS_LR          = 1e-3
CLS_WEIGHT_DECAY= 1e-5
CLS_DROPOUT     = 0.10
CLS_HIDDEN      = 128
CLS_SAVE_PATH   = os.path.join(OUT_DIR, "day_surge_classifier.pt")
# NEW: label smoothing
CLS_LABEL_SMOOTH = 0.05

# ======================== NEW: training stabilizers ========================
# Cosine after warmup (fixes early-epoch plateau/overshoot)
WARMUP_EPOCHS = 3
MIN_LR        = 1e-5

# EMA for evaluation (stabilizes validation curve)
USE_EMA    = True
EMA_DECAY  = 0.999

# =======================================================
# utils
# =======================================================
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

# =======================================================
# encoders (PatchTST / Chronos-Bolt PEFT)
# =======================================================
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
        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        L = (T + S - 1) // S
        T_pad = (L - 1) * S + P
        if T_pad < T:
            L += 1
            T_pad = (L - 1) * S + P
        if T_pad > T:
            pad = torch.zeros(B * N, T_pad - T, F, device=x.device, dtype=x.dtype)
            x_bn = torch.cat([x_bn, pad], dim=1)
        x_unf = x_bn.unfold(dimension=1, size=P, step=S)
        x_unf = x_unf.contiguous().view(B * N, L, P * F)
        tokens = self.proj(x_unf)
        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)
        tokens_rep = tokens.repeat_interleave(P, dim=1)
        steps = tokens_rep[:, :T, :]
        h = steps.view(B, N, T, self.d_model).permute(0, 2, 1, 3)
        return h

def _discover_lora_targets(model: nn.Module):
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    candidates = [
        "q_proj","k_proj","v_proj","o_proj","query_key_value","out_proj","in_proj",
        "to_q","to_k","to_v","to_out","proj_in","proj_out",
        "q","k","v","o","wi","wo","wi_0","wi_1","Wqkv","Wo","Wq","Wk","Wv",
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
                "Install deps: pip install transformers peft accelerate safetensors\n"
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
        from peft import get_peft_model
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
            out = self.backbone.encoder(inputs_embeds=emb, attention_mask=attn_mask, output_hidden_states=True)
            h_bn = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        else:
            out = self.backbone(inputs_embeds=emb, attention_mask=attn_mask, output_hidden_states=True)
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

# =======================================================
# StemGNN blocks
# =======================================================
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

# =======================================================
# MoE readout
# =======================================================
class MoEReadout(nn.Module):
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
        B, N, D = h_last.shape
        logits = self.router(h_last)
        probs  = torch.softmax(logits, dim=-1)
        if self.top_k < self.num_experts:
            topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
            norm = torch.clamp(topv.sum(dim=-1, keepdim=True), min=1e-9)
            topw = topv / norm
        else:
            topi = torch.arange(self.num_experts, device=h_last.device).view(1,1,self.num_experts).expand(B,N,self.num_experts)
            topw = probs
        expert_outs = torch.stack([exp(h_last) for exp in self.experts], dim=-1)
        gather_idx = topi.unsqueeze(-2).expand(B, N, expert_outs.size(-2), topw.size(-1))
        selected = expert_outs.gather(dim=-1, index=gather_idx)
        y = (selected * topw.unsqueeze(-2)).sum(dim=-1)
        probs_mean = probs.mean(dim=(0,1))
        eps = 1e-9
        aux = (probs_mean * (torch.log(probs_mean + eps) + math.log(self.num_experts))).sum()
        return y, aux, probs_mean

# =======================================================
# Forecaster
# =======================================================
class STGNNForecaster(nn.Module):
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels
        self.n_quantiles = len(QUANTILES) if USE_QUANTILES else 1
        out_dim = horizon * self.n_quantiles

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

        if USE_MOE:
            self.readout_moe = MoEReadout(
                d_in=STEM_D_MODEL, out_dim=out_dim,
                num_experts=MOE_NUM_EXPERTS, hidden=MOE_HIDDEN,
                top_k=MOE_TOP_K, dropout=MOE_DROPOUT
            )
            self.readout = None
            print(f"[MoE] Enabled: E={MOE_NUM_EXPERTS}, top_k={MOE_TOP_K}, hidden={MOE_HIDDEN}, out_dim={out_dim}")
        else:
            self.readout = nn.Linear(STEM_D_MODEL, out_dim)
            self.readout_moe = None

        self._last_aux_loss = None
        self._last_gate_mean = None

    def forward(self, x):
        B, T, N, F = x.shape
        assert N == self.num_nodes and F == self.in_channels, \
            f"StemGNN input shape mismatch: got N={N},F={F}, expected N={self.num_nodes},F={self.in_channels}"
        h = self.patch(x)
        for blk in self.blocks:
            h = blk(h)
        last = h[:, -1]
        if self.readout_moe is not None:
            y_flat, aux, probs_mean = self.readout_moe(last)
            self._last_aux_loss = aux
            self._last_gate_mean = probs_mean.detach()
        else:
            y_flat = self.readout(last)
            self._last_aux_loss = None
            self._last_gate_mean = None
        if USE_QUANTILES:
            y = y_flat.view(B, N, self.horizon, len(QUANTILES))
            return y
        else:
            return y_flat
    def moe_aux_loss(self):
        return self._last_aux_loss
    def moe_gate_usage(self):
        return self._last_gate_mean

# =======================================================
# Day-ahead surge classifier
# =======================================================
class DaySurgeClassifier(nn.Module):
    def __init__(self, d_in, hidden=CLS_HIDDEN, dropout=CLS_DROPOUT, n_classes=3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(d_in, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.temporal(x)
        logits = self.head(h)
        return logits

# =======================================================
# helpers for day split & scalers
# =======================================================
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

# =======================================================
# labels for surge classifier
# =======================================================
def _levels_from_csv(csv_path, max_day):
    if not (USE_SURGE_RANK_CSV and os.path.exists(csv_path)):
        return None
    df = pd.read_csv(csv_path)
    if "day_index" not in df.columns or "level" not in df.columns:
        return None
    df = df[df["day_index"] <= int(max_day)].copy()
    def map_level(s):
        s = str(s).strip().lower()
        if s in ("calm", "mild"):
            return 0
        if s == "moderate":
            return 1
        if s in ("severe", "extreme", "storm"):
            return 2
        return 1
    df["label"] = df["level"].map(map_level)
    return {int(r.day_index): int(r.label) for r in df.itertuples(index=False)}

def _levels_from_data(ds_train):
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    ts = ds_train.timestamp.values
    days, uniq, _ = _compute_day_indices(ts)
    per_day = []
    for i, dts in enumerate(uniq, start=1):
        x = y[days == i].mean(axis=1)
        per_day.append([i, float(np.nanmax(x))])
    df = pd.DataFrame(per_day, columns=["day_index", "day_max"])
    q = df["day_max"].quantile([0.25, 0.75]).values
    q25, q75 = float(q[0]), float(q[1])
    labels = {}
    for r in df.itertuples(index=False):
        if r.day_max <= q25:
            labels[int(r.day_index)] = 0
        elif r.day_max >= q75:
            labels[int(r.day_index)] = 2
        else:
            labels[int(r.day_index)] = 1
    return labels

# =======================================================
# build day-level classifier dataset
# =======================================================
def build_day_cls_dataset(ds_train, scalers, seq_len=SEQ_LEN, train_last_day=TRAIN_LAST_DAY):
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, uniq_days, day_end_idx = _compute_day_indices(ts)
    T, N, F = w.shape
    Din = 1 + F
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)
    lab_map = _levels_from_csv(SURGE_CSV_PATH, max_day=int(days.max()))
    if lab_map is None:
        lab_map = _levels_from_data(ds_train)
    X_list, y_list, d_src = [], [], []
    for d in range(1, train_last_day):
        if (d+1) not in lab_map:
            continue
        end_idx = day_end_idx[d - 1]
        start_ctx = end_idx - (seq_len - 1)
        if start_ctx < 0:
            continue
        x_ctx = X_all[start_ctx:end_idx+1].mean(axis=1)  # (SEQ_LEN, Din)
        X_list.append(x_ctx.astype(np.float32))
        y_list.append(int(lab_map[d+1]))
        d_src.append(d)
    if not X_list:
        raise RuntimeError("No day-level classifier samples created; check data and labels.")
    X_cls = np.stack(X_list, axis=0)
    y_cls = np.array(y_list, dtype=np.int64)
    return X_cls, y_cls, Din, {"days": d_src, "lab_map": lab_map}

# =======================================================
# train classifier  (NEW: label_smoothing)
# =======================================================
def train_day_classifier(X_cls, y_cls, Din_agg):
    cls = DaySurgeClassifier(d_in=Din_agg).to(DEVICE)
    ds = TensorDataset(torch.tensor(X_cls, dtype=torch.float32), torch.tensor(y_cls, dtype=torch.long))
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    uniq, counts = np.unique(y_cls, return_counts=True)
    cw = np.zeros(3, dtype=np.float32) + 1.0
    for u, c in zip(uniq, counts):
        cw[int(u)] = float(len(y_cls) / (3.0 * max(1, c)))
    ce = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, device=DEVICE, dtype=torch.float32),
        label_smoothing=CLS_LABEL_SMOOTH  # NEW
    )
    opt = torch.optim.Adam(cls.parameters(), lr=CLS_LR, weight_decay=CLS_WEIGHT_DECAY)

    for ep in range(CLS_EPOCHS):
        cls.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = cls(xb)
            loss = ce(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(cls.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            tot += xb.size(0)
        print(f"[DayCLS] Epoch {ep+1}/{CLS_EPOCHS} | loss={loss_sum/tot:.4f} | acc={correct/tot:.3f}")

    torch.save({"state_dict": cls.state_dict(), "Din_agg": int(Din_agg)}, CLS_SAVE_PATH)
    print(f"[DayCLS] Saved: {CLS_SAVE_PATH}")
    return cls

@torch.no_grad()
def classify_day(cls_model, x_ctx_agg):
    cls_model.eval()
    xb = torch.tensor(x_ctx_agg, dtype=torch.float32, device=DEVICE)
    logits = cls_model(xb)  # (1,3)
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    return probs  # [p_calm, p_moderate, p_storm]

def blend_quantile_weights(probs):
    w_calm = REGIME_QWEIGHTS["calm"]
    w_mod  = REGIME_QWEIGHTS["moderate"]
    w_stm  = REGIME_QWEIGHTS["storm"]
    qw = probs[0]*w_calm + probs[1]*w_mod + probs[2]*w_stm
    qw = qw / max(1e-8, qw.sum())
    return qw.astype(np.float32)

# =======================================================
# training split for forecaster
# =======================================================
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

# =======================================================
# validation roll with dynamic blending
# =======================================================
@torch.no_grad()
def predict_val_roll(ds_train, model, cls_model, scalers, horizon, aux, train_last_day=80):
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
        ctx_agg = X_all[start_ctx:end_idx+1].mean(axis=1)
        probs = classify_day(cls_model, ctx_agg[None, ...])
        qw = blend_quantile_weights(probs)
        for i, county in enumerate(locs):
            Xin_i = X_all[start_ctx:end_idx+1, i, :].reshape(1, SEQ_LEN, 1, X_all.shape[2])
            yq_sc = infer_quantiles(model, Xin_i)[0, 0]  # (H, Q)
            pred_sc = (yq_sc * qw[None, :]).sum(axis=1)
            y_mu = float(scalers["y_mu"].flatten()[0])
            y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0
            pred = (pred_sc * y_sd) + y_mu
            pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)
            rows.append(pd.DataFrame({
                "timestamp": ts_future,
                "location": county,
                "pred": pred.astype(float)
            }))
    df_pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp", "location", "pred"])
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
    pw = df_p.pivot_table(index="timestamp", columns="location", values="pred", aggfunc="mean").sort_index()
    common_ts = tw.index.intersection(pw.index)
    if len(common_ts) == 0:
        return float("nan"), float("nan")
    tw = tw.loc[common_ts]
    pw = pw.loc[common_ts]
    locs = sorted(set(tw.columns).intersection(pw.columns))
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

def make_val_callback_rolling(ds_train, cls_model, scalers, horizon, aux, label,
                              train_last_day=80,
                              ts_future_test=None, df_truth_test=None, zero_rmse_test=float("nan")):
    best = {"rmse": float("inf"), "epoch": -1}
    df_truth_val = truth_wide_for_val(ds_train, aux, horizon, train_last_day)
    def _callback(model, epoch):
        df_pred_val = predict_val_roll(ds_train, model, cls_model, scalers, horizon, aux, train_last_day)
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

        print(f"[Val {label}] Epoch {epoch}: Zero→Nonzero={zpnz_model if np.isfinite(zpnz_model) else float('nan'):.4f} "
              f"(all-zero={zpnz_zero if np.isfinite(zpnz_zero) else float('nan'):.4f}) | "
              f"Nonzero→Zero={nzpz_model if np.isfinite(nzpz_model) else float('nan'):.4f} "
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

    return _callback

# =======================================================
# quantile-aware inference helpers
# =======================================================
@torch.no_grad()
def infer_quantiles(model, Xin):
    model.eval()
    out = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE))
    if USE_QUANTILES:
        return out.cpu().numpy()
    else:
        y = out.cpu().numpy()
        return y[..., None]

@torch.no_grad()
def infer_point_with_blend(model, Xin, qw, pred_quantile=PRED_QUANTILE):
    yq = infer_quantiles(model, Xin)  # (1,N,H,Q)
    if yq.shape[-1] >= 2:
        qw = np.asarray(qw, dtype=np.float32)
        qw = qw / max(qw.sum(), 1e-8)
        y = (yq * qw.reshape(1,1,1,-1)).sum(axis=-1)
    else:
        y = yq[..., 0]
    return y

# =======================================================
# training / inference for forecaster (pinball)
# =======================================================
def _pinball_loss(yq, y_true, q_list):
    qs = torch.tensor(q_list, device=yq.device, dtype=yq.dtype).view(1,1,1,-1)
    diff = y_true.unsqueeze(-1) - yq
    loss = torch.maximum(qs * diff, (qs - 1.0) * diff)
    return loss.mean()

# NEW: simple EMA utility
class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()
        self.backup = {}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
    @torch.no_grad()
    def swap_to_shadow(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
    @torch.no_grad()
    def swap_back(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

def train_model(X, Y, din, horizon, n_nodes, eval_callback=None):
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

    # NEW: LR schedule = warmup -> cosine
    sched = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=0.1, total_iters=WARMUP_EPOCHS),
            CosineAnnealingLR(opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=MIN_LR),
        ],
        milestones=[WARMUP_EPOCHS]
    )

    # NEW: EMA
    ema = _EMA(model, decay=EMA_DECAY) if USE_EMA else None

    loss_fn_point = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if TEMPORAL_DROP_PROB > 0.0:
                with torch.no_grad():
                    drop_mask = (torch.rand(xb.size(0), xb.size(1), 1, 1, device=xb.device) < TEMPORAL_DROP_PROB).float()
                    xb = xb * (1.0 - drop_mask)

            opt.zero_grad()
            pred = model(xb)  # (B,N,H,Q) or (B,N,H)
            if USE_QUANTILES:
                loss = _pinball_loss(pred, yb, QUANTILES)
            else:
                loss = loss_fn_point(pred, yb)

            aux = model.moe_aux_loss()
            if aux is not None:
                loss = loss + W_MOE_AUX * aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            # NEW: update EMA each step
            if ema is not None:
                ema.update(model)

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

        # NEW: evaluate with EMA weights swapped in
        if eval_callback is not None:
            if ema is not None:
                ema.swap_to_shadow(model)
                eval_callback(model, epoch + 1)
                ema.swap_back(model)
            else:
                eval_callback(model, epoch + 1)

        # NEW: step scheduler per epoch
        sched.step()

    return model

# =======================================================
# test-time prediction (dynamic blending using classifier)
# =======================================================
def predict_per_county(ds_train, model, cls_model, scalers, horizon, ts_future):
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, _, day_end_idx = _compute_day_indices(ts)
    T, N, F = w.shape
    Din = 1 + F
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)
    end_idx = day_end_idx[TRAIN_LAST_DAY - 1]
    start_ctx = max(0, end_idx - (SEQ_LEN - 1))
    ctx_agg = X_all[start_ctx:end_idx+1].mean(axis=1)
    probs = classify_day(cls_model, ctx_agg[None, ...])
    qw = blend_quantile_weights(probs)
    rows = []
    for i, county in enumerate(locs):
        if T < SEQ_LEN:
            pred_sc = np.zeros(horizon, dtype=float)
        else:
            Xin_i = X_all[-SEQ_LEN:, i, :].reshape(1, SEQ_LEN, 1, Din)
            yq_sc = infer_quantiles(model, Xin_i)[0, 0]
            pred_sc = (yq_sc * qw[None, :]).sum(axis=1)
        pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)
        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))
    return pd.concat(rows, ignore_index=True)

# =======================================================
# main
# =======================================================
def main():
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

    # ---- Prepare scalers once
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, _, _ = _compute_day_indices(ts)
    scalers = _fit_scalers_train_only(y, w, days, TRAIN_LAST_DAY)

    # ---- Train day-level classifier
    X_cls, y_cls, Din_agg, _ = build_day_cls_dataset(ds_train, scalers, seq_len=SEQ_LEN, train_last_day=TRAIN_LAST_DAY)
    cls_model = train_day_classifier(X_cls, y_cls, Din_agg)

    # ==== 24h forecaster ====
    X24, Y24, Din, scalers24, aux24 = prepare_training_split(ds_train, horizon=H24, train_last_day=TRAIN_LAST_DAY)
    n_nodes = 1
    val_cb24 = make_val_callback_rolling(
        ds_train, cls_model, scalers24, H24, aux24, label="24h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts24, df_truth_test=df_t24, zero_rmse_test=zero_rmse_24
    )
    mdl24 = train_model(X24, Y24, Din, H24, n_nodes, eval_callback=val_cb24)

    # ==== 48h forecaster ====
    X48, Y48, _, scalers48, aux48 = prepare_training_split(ds_train, horizon=H48, train_last_day=TRAIN_LAST_DAY)
    val_cb48 = make_val_callback_rolling(
        ds_train, cls_model, scalers48, H48, aux48, label="48h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts48, df_truth_test=df_t48, zero_rmse_test=zero_rmse_48
    )
    mdl48 = train_model(X48, Y48, Din, H48, n_nodes, eval_callback=val_cb48)

    # ----- Test evaluation using best models, then write CSVs -----
    if mdl24 is not None:
        df24 = predict_per_county(ds_train, mdl24, cls_model, scalers24, H24, ts24)
    else:
        df24 = None
    if mdl48 is not None:
        df48 = predict_per_county(ds_train, mdl48, cls_model, scalers48, H48, ts48)
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
