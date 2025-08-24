import os
import time
from math import ceil
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =========================
# Config
# =========================
SPLIT_DIR   = "data/"
OUT_DIR     = "results_lora_new/"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# Windows / data
SEQ_LEN     = 24     # context length T (match downstream)
H_PRE       = 24     # pretraining forecast horizon H
BATCH_SIZE  = 256   # batch size for training

# Training
EPOCHS          = 100
LR              = 1e-4
WEIGHT_DECAY    = 1e-3
GRAD_CLIP_NORM  = 5.0

# ---------- Masking for SSL ----------
MASK_MODE               = "span"   # "random" or "span"
MASK_RATIO              = 0.15     # fraction of timesteps masked per window (lighter masking for better context)
SPAN_LEN                = 4        # used if MASK_MODE=="span": contiguous length per span

# For contrastive views (lighter masking by default)
VIEW_MASK_MODE          = "span"   # "random" or "span"
VIEW_MASK_RATIO         = 0.15
VIEW_SPAN_LEN           = 3

# ---------- Contrastive (InfoNCE) ----------
USE_CONTRASTIVE         = True
W_CONTRAST              = 0.5
TAU                     = 0.2
ADD_JITTER_STD          = 0.01
ADD_DROPOUT_PROB        = 0.0
CONTRAST_UPDATES_ENCODER = False

# ---------- Frequency-domain augmentations ----------
USE_FREQ_AUG            = True
FREQ_AUG_PROB           = 0.8
FREQ_AMP_STD            = 0.10
FREQ_PHASE_STD          = 0.20
FREQ_DROPOUT_FRAC       = 0.10
FREQ_DROPOUT_PROB       = 0.5
FREQ_BAND_PROB          = 0.5
FREQ_BAND_WIDTH_RATIO   = (0.05, 0.25)
FREQ_SHELF_PROB         = 0.3
FREQ_SHELF_STRENGTH     = (0.3, 0.8)

# ---------- Multitask loss weights ----------
W_SSL       = 1.0
W_FORECAST  = 1.0

# ---------- CB + PEFT (LoRA) ----------
CB_MODEL_NAME   = "/home2/caojia/weather_prediction/code/chronos-bolt-small"
D_OUT           = 64
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

# ---------- Train/Val split by days ----------
TRAIN_LAST_DAY = 80

# ---------- Event-stratified validation ----------
EVENT_METHOD   = "mean"
STORM_TOP_Q    = 0.80
CALM_BOTTOM_Q  = 0.50
CALM_N         = 10
BEST_BY        = "overall"  # "overall" | "calm" | "storm" | "loss"


# Save paths
SAVE_PATH = os.path.join(OUT_DIR, f"pretrained_cb_encoder_peft_independent_H{H_PRE}.pt")
SAVE_ADAPTER_DIR = os.path.join(OUT_DIR, f"cb_peft_adapter_independent_H{H_PRE}")

# =========================
# Utils
# =========================
def open_ds(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path, engine="netcdf4")

def zfit(arr):
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def zapply(arr, mu, sd):
    return (arr - mu) / sd

# =========================
# Data prep (independent locations)
# =========================
def prepare_multitask_windows_split(ds_train, seq_len=24, horizon=24, train_last_day=80):
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, uniq_days, day_end_idx = _compute_day_indices(ts)

    T, N, F = w.shape
    Din = 1 + F

    scalers = _fit_scalers_train_only(y, w, days, train_last_day)
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)

    nW_total = T - seq_len - horizon + 1
    if nW_total <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    last_train_idx = day_end_idx[train_last_day - 1]
    nW_train = max(0, last_train_idx - seq_len - horizon + 1)

    X_list, Y_list = [], []
    for node in range(N):
        x_loc = X_all[:, node, :]
        Xw = np.stack([x_loc[i:i+seq_len] for i in range(nW_total)], axis=0)
        Yw = np.stack([X_all[i+seq_len:i+seq_len+horizon, node, :] for i in range(nW_total)], axis=0)
        if nW_train > 0:
            X_list.append(Xw[:nW_train][:, :, None, :])
            Y_list.append(Yw[:nW_train][:, :, None, :])

    if not X_list:
        X_tr = np.empty((0, seq_len, 1, Din), dtype=float)
        Y_tr = np.empty((0, horizon, 1, Din), dtype=float)
    else:
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

def build_val_windows_day_ahead(aux, horizon, seq_len, train_last_day=80):
    X_all = aux["X_all_scaled"]
    ts    = aux["timestamps"]
    days  = aux["days"]
    day_end_idx = aux["day_end_idx"]
    T, N, Din = X_all.shape

    Xv, Yv = [], []
    last_day = int(days.max())
    for d in range(train_last_day, last_day):
        end_idx = day_end_idx[d - 1]
        start_ctx = end_idx - (seq_len - 1)
        fut_start = end_idx + 1
        fut_end   = fut_start + horizon - 1
        if start_ctx < 0 or fut_end >= T:
            continue
        for node in range(N):
            ctx = X_all[start_ctx:end_idx+1, node, :].reshape(seq_len, 1, Din)
            fut = X_all[fut_start:fut_start+horizon, node, :].reshape(horizon, 1, Din)
            Xv.append(ctx)
            Yv.append(fut)

    if not Xv:
        return (np.empty((0, seq_len, 1, Din), dtype=float),
                np.empty((0, horizon, 1, Din), dtype=float))
    Xv = np.stack(Xv, axis=0)
    Yv = np.stack(Yv, axis=0)
    return Xv, Yv

# =========================
# Split helpers
# =========================
def _compute_day_indices(ts_np: np.ndarray):
    ts = pd.to_datetime(ts_np)
    days_norm = ts.normalize()
    uniq = pd.DatetimeIndex(sorted(days_norm.unique()))
    day_map = {d: i + 1 for i, d in enumerate(uniq)}
    days = np.array([day_map[d] for d in days_norm], dtype=int)
    day_end_idx = []
    for d in uniq:
        idxs = np.where(days_norm == d)[0]
        day_end_idx.append(int(idxs.max()))
    return days, list(uniq), day_end_idx

def _fit_scalers_train_only(y, w, days, train_last_day):
    mask = days <= train_last_day
    y_tr = y[mask, :].reshape(-1, 1)
    w_tr = w[mask, :, :].reshape(-1, w.shape[2])
    y_mu, y_sd = zfit(y_tr)
    w_mu, w_sd = zfit(w_tr)
    return {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}

# =========================
# Chronos-Bolt PEFT encoder
# =========================
try:
    from transformers import AutoModel, AutoConfig
    HAVE_TRANSFORMERS = True
except Exception as e:
    HAVE_TRANSFORMERS = False
    _transformers_err = str(e)

try:
    from peft import LoraConfig, get_peft_model
    HAVE_PEFT = True
except Exception as e:
    HAVE_PEFT = False
    _peft_err = str(e)

def _discover_lora_targets(model: nn.Module):
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "query_key_value", "out_proj", "in_proj",
        "to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out",
        "q", "k", "v", "o", "wi", "wo", "wi_0", "wi_1",
        "Wqkv", "Wo", "Wq", "Wk", "Wv",
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
    def __init__(self, in_channels, cb_model_name=CB_MODEL_NAME, d_out=D_OUT,
                 lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT):
        super().__init__()
        if not HAVE_TRANSFORMERS:
            raise ImportError(f"transformers not available: {_transformers_err}")
        if not HAVE_PEFT:
            raise ImportError(f"peft not available: {_peft_err}")

        self.in_channels = in_channels
        self.d_out = d_out

        self.config = AutoConfig.from_pretrained(cb_model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            cb_model_name, trust_remote_code=True, output_hidden_states=True
        )

        targets = _discover_lora_targets(self.backbone)
        print(f"[PEFT] LoRA targets: {targets}")

        lcfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=targets, bias="none", task_type="FEATURE_EXTRACTION",
        )
        try:
            self.backbone = get_peft_model(self.backbone, lcfg)
        except ValueError as e:
            print(f"[PEFT] Warning: {e} -> fallback target_modules=['linear']")
            lcfg.target_modules = ["linear"]
            self.backbone = get_peft_model(self.backbone, lcfg)

        hidden_dim = getattr(self.config, "hidden_size", None) or getattr(self.config, "d_model", None)
        if hidden_dim is None:
            raise RuntimeError("Cannot determine Chronos hidden size from config.")

        self.embed_proj = nn.Linear(in_channels, hidden_dim)
        self.post_proj  = nn.Linear(hidden_dim, d_out)

    def forward(self, x):
        B, T, N, F = x.shape
        assert F == self.in_channels, f"CB encoder expected F={self.in_channels}, got {F}"

        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        emb  = self.embed_proj(x_bn)
        attn_mask = torch.ones(B * N, T, dtype=torch.long, device=emb.device)

        if hasattr(self.backbone, "encoder") and callable(self.backbone.encoder):
            out = self.backbone.encoder(inputs_embeds=emb, attention_mask=attn_mask, output_hidden_states=True)
        else:
            out = self.backbone(inputs_embeds=emb, attention_mask=attn_mask, output_hidden_states=True)

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            h_bn = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            h_bn = out.hidden_states[-1]
        else:
            if isinstance(out, (tuple, list)) and len(out) > 0:
                h_bn = out[0]
            else:
                raise RuntimeError("Backbone forward did not return hidden states.")

        if h_bn.size(1) > T:
            h_bn = h_bn[:, :T, :]
        elif h_bn.size(1) < T:
            pad_len = T - h_bn.size(1)
            pad = torch.zeros(h_bn.size(0), pad_len, h_bn.size(2), device=h_bn.device, dtype=h_bn.dtype)
            h_bn = torch.cat([h_bn, pad], dim=1)

        h_bn = self.post_proj(h_bn)
        h = h_bn.view(B, N, T, self.d_out).permute(0, 2, 1, 3)
        return h

    def print_trainable(self):
        try:
            self.backbone.print_trainable_parameters()
        except Exception:
            total, trainable = 0, 0
            for p in self.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            print(f"[PEFT] Trainable params {trainable}/{total} ({100.0*trainable/total:.2f}%)")

# =========================
# Multi-task model using CB encoder
# =========================
class MultiTaskCBEncoder(nn.Module):
    def __init__(self, in_channels, horizon):
        super().__init__()
        self.horizon = horizon
        self.encoder = ChronosBoltEncoderPEFT(in_channels=in_channels, d_out=D_OUT)

        self.recon_head = nn.Linear(D_OUT, in_channels)
        self.forecast_head = nn.Sequential(
            nn.Linear(D_OUT, 2 * D_OUT),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * D_OUT, horizon * in_channels)
        )
        self.proj_head = nn.Sequential(
            nn.Linear(D_OUT, D_OUT),
            nn.GELU(),
            nn.Linear(D_OUT, D_OUT)
        )

    def encode_steps(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)                    # (B, T, N, D_OUT)
        recon = self.recon_head(h)             # (B, T, N, F)
        pooled = h.mean(dim=1)                 # (B, N, D_OUT)
        fut = self.forecast_head(pooled)       # (B, N, H*F)
        B, N, _ = fut.shape
        fut = fut.view(B, N, self.horizon, -1).permute(0, 2, 1, 3)  # (B, H, N, F)
        return recon, fut

    def pooled_projection(self, x):
        h = self.encoder(x) if CONTRAST_UPDATES_ENCODER else self.encoder(x).detach()
        pooled = h.mean(dim=1).mean(dim=1)       # (B, D_OUT)
        z = self.proj_head(pooled)               # (B, D_OUT)
        z = F.normalize(z, dim=-1)
        return z

# =========================
# Losses
# =========================
def masked_recon_loss(pred, target, mask):
    m = mask.expand_as(pred)
    denom = m.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * m).sum() / denom

def forecast_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def info_nce_loss(z1, z2, tau=0.2):
    B = z1.size(0)
    if B == 1:
        return torch.tensor(0.0, device=z1.device)
    sim = (z1 @ z2.t()) / tau
    labels = torch.arange(B, device=z1.device)
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_12 + loss_21)

# =========================
# Masking & Augmentations
# =========================
def make_temporal_mask(batch_size, T, mode="random", ratio=0.3, span_len=4, device="cpu"):
    if mode == "random":
        m = (torch.rand(batch_size, T, 1, 1, device=device) < ratio).float()
        return m
    K = max(1, ceil((ratio * T) / max(1, span_len)))
    m = torch.zeros(batch_size, T, 1, 1, device=device)
    for b in range(batch_size):
        for _ in range(K):
            if T <= span_len:
                start = 0
            else:
                start = torch.randint(0, T - span_len + 1, (1,), device=device).item()
            m[b, start:start+span_len, :, :] = 1.0
    return m.clamp(max=1.0)

def apply_mask(x, mask):
    return x * (1.0 - mask)

def add_jitter(x, std=0.01):
    if std <= 0.0:
        return x
    noise = torch.randn_like(x) * std
    return x + noise

def dropout_features(x, p=0.0):
    if p <= 0.0:
        return x
    B, T, N, F = x.shape
    mask = (torch.rand(B, 1, 1, F, device=x.device) < p).float()
    return x * (1.0 - mask)

@torch.no_grad()
def _variance_match(x_aug, x_ref, eps=1e-6):
    s_ref = x_ref.flatten(1).std(dim=1, keepdim=True).clamp_min(eps)
    s_aug = x_aug.flatten(1).std(dim=1, keepdim=True).clamp_min(eps)
    scale = (s_ref / s_aug).view(-1, 1, 1, 1)
    return x_aug * scale

def _random_band_mask(B, Lf, device, keep=False):
    centers = torch.rand(B, device=device) * (Lf - 1)
    widths  = torch.empty(B, device=device).uniform_(
        int(FREQ_BAND_WIDTH_RATIO[0] * Lf),
        max(1.0, FREQ_BAND_WIDTH_RATIO[1] * Lf)
    )
    idx = torch.arange(Lf, device=device).view(1, Lf)
    mask = torch.ones(B, Lf, 1, 1, device=device)
    for b in range(B):
        c = centers[b]
        w = widths[b]
        lo = (c - w / 2.0).clamp(0, Lf - 1)
        hi = (c + w / 2.0).clamp(0, Lf - 1)
        band = ((idx >= lo) & (idx <= hi)).float().view(1, Lf, 1, 1)
        mask[b:b+1] = band if keep else (1.0 - band)
    return mask

def _shelf_curve(Lf, high=True, strength=0.5, device="cpu"):
    f = torch.linspace(0.0, 1.0, steps=Lf, device=device).view(1, Lf, 1, 1)
    if high:
        curve = 1.0 - strength * f.pow(1.0)
    else:
        curve = 1.0 - strength * (1.0 - f).pow(1.0)
    return curve.clamp_min(0.0)

def freq_augment(x, prob=FREQ_AUG_PROB):
    if (not USE_FREQ_AUG) or (torch.rand(1).item() > prob):
        return x
    B, T, N, Fdim = x.shape
    Xf = torch.fft.rfft(x, dim=1)
    Lf = Xf.size(1)
    if FREQ_AMP_STD > 0.0:
        amp = torch.randn(B, Lf, 1, 1, device=x.device, dtype=Xf.real.dtype) * FREQ_AMP_STD + 1.0
        Xf = Xf * amp
    if FREQ_PHASE_STD > 0.0:
        phi = torch.randn(B, Lf, 1, 1, device=x.device, dtype=Xf.real.dtype) * FREQ_PHASE_STD
        rot = torch.complex(torch.cos(phi), torch.sin(phi))
        Xf = Xf * rot
    keep = not (torch.rand(1).item() < FREQ_BAND_PROB)
    band_mask = _random_band_mask(B, Lf, device=x.device, keep=keep)
    Xf = Xf * band_mask
    if torch.rand(1).item() < FREQ_DROPOUT_PROB and FREQ_DROPOUT_FRAC > 0.0:
        k = max(1, int(FREQ_DROPOUT_FRAC * Lf))
        drop = torch.zeros(B, Lf, 1, 1, device=x.device)
        for b in range(B):
            idx = torch.randperm(Lf, device=x.device)[:k]
            drop[b, idx] = 1.0
        Xf = Xf * (1.0 - drop)
    if torch.rand(1).item() < FREQ_SHELF_PROB:
        strength = torch.empty(1, device=x.device).uniform_(*FREQ_SHELF_STRENGTH).item()
        high = bool(torch.randint(0, 2, (1,), device=x.device).item())
        shelf = _shelf_curve(Lf, high=high, strength=strength, device=x.device)
        Xf = Xf * shelf
    x_aug = torch.fft.irfft(Xf, n=T, dim=1)
    x_aug = _variance_match(x_aug, x_ref=x)
    return x_aug

def make_view(x, mask_mode, mask_ratio, span_len, jitter_std, feat_drop):
    B, T, _, _ = x.shape
    m = make_temporal_mask(B, T, mode=mask_mode, ratio=mask_ratio, span_len=span_len, device=x.device)
    xv = apply_mask(x, m)
    xv = dropout_features(xv, p=feat_drop)
    xv = add_jitter(xv, std=jitter_std)
    xv = freq_augment(xv, prob=FREQ_AUG_PROB)
    return xv

# =========================
# Small eval helpers for rolling validation
# =========================
def _pivot_pred_long_to_wide(df_long: pd.DataFrame, locs: list[str]) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=["timestamp", *locs])
    pivot = df_long.pivot_table(index="timestamp", columns="location", values="pred", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(columns=locs, fill_value=0.0).sort_index().reset_index()
    pivot.columns.name = None
    return pivot

def _per_county_rmse_mean(df_truth_wide: pd.DataFrame, df_pred_long: pd.DataFrame) -> float:
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]
    df_pred_wide = _pivot_pred_long_to_wide(df_pred_long, locs)
    df = df_truth_wide.merge(df_pred_wide, on="timestamp", how="inner", suffixes=("_true", "_pred"))
    if df.empty:
        return float("nan")
    rmses = []
    for loc in locs:
        yt = df[f"{loc}_true"].values.astype(float)
        yp = df[f"{loc}_pred"].values.astype(float) if f"{loc}_pred" in df.columns else np.zeros_like(yt)
        rmses.append(float(np.sqrt(np.mean((yt - yp) ** 2))))
    return float(np.mean(rmses)) if rmses else float("nan")

def _all_zero_rmse_for_truth(df_truth_wide: pd.DataFrame) -> float:
    if df_truth_wide.empty:
        return float("nan")
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]
    df_zero_long = pd.DataFrame({
        "timestamp": np.repeat(df_truth_wide["timestamp"].values, len(locs)),
        "location": np.tile(locs, len(df_truth_wide)),
        "pred": 0.0
    })
    return _per_county_rmse_mean(df_truth_wide, df_zero_long)

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
    mask = np.array([(d in select_days) for d in days], dtype=bool)
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
    rmse_overall = _per_county_rmse_mean(df_truth_val, df_pred_val) if (not df_truth_val.empty and not df_pred_val.empty) else float("nan")
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
        rmse_calm = _per_county_rmse_mean(truth_calm, pred_calm)
        zero_calm = _all_zero_rmse_for_truth(truth_calm)
        n_calm = len(calm_days)
    else:
        rmse_calm, zero_calm, n_calm = float("nan"), float("nan"), 0

    if len(ts_storm) > 0:
        truth_storm, pred_storm = _filter_truth_pred_by_days(df_truth_val, df_pred_val, ts_storm)
        rmse_storm = _per_county_rmse_mean(truth_storm, pred_storm)
        zero_storm = _all_zero_rmse_for_truth(truth_storm)
        n_storm = len(storm_days)
    else:
        rmse_storm, zero_storm, n_storm = float("nan"), float("nan"), 0

    return rmse_overall, zero_overall, rmse_calm, zero_calm, n_calm, rmse_storm, zero_storm, n_storm

@torch.no_grad()
def _truth_wide_for_val(ds_train, aux, horizon, train_last_day=80) -> pd.DataFrame:
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

@torch.no_grad()
def _predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day=80) -> pd.DataFrame:
    """
    Rolling validation predictions with independent per-county inference (N=1).
    Returns long df [timestamp, location, pred].
    """
    model.eval()
    X_all = aux["X_all_scaled"]
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
        if start_ctx < 0 or fut_start + horizon - 1 >= len(ts):
            continue

        ts_future = pd.to_datetime(ts[fut_start:fut_start + horizon])

        for i, county in enumerate(locs):
            Xin_i = X_all[start_ctx:end_idx+1, i, :].reshape(1, SEQ_LEN, 1, Din)  # (1, T, 1, Din)
            xb = torch.tensor(Xin_i, dtype=torch.float32, device=DEVICE)
            _, fut_i = model(xb)                              # <— clean context inference
            fut_np = fut_i[0, :, 0, :].detach().cpu().numpy() # (H, F)
            pred_sc = fut_np[:, 0]                            # outage channel

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
        return pd.concat(rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=["timestamp", "location", "pred"])

# =========================
# Training loop
# =========================
def train_multitask_contrast(ds_train):
    X_tr, Y_fut_tr, Din, scalers, aux = prepare_multitask_windows_split(
        ds_train, seq_len=SEQ_LEN, horizon=H_PRE, train_last_day=TRAIN_LAST_DAY
    )
    dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(Y_fut_tr, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_val, Y_val = build_val_windows_day_ahead(aux, horizon=H_PRE, seq_len=SEQ_LEN, train_last_day=TRAIN_LAST_DAY)
    have_val_loss = X_val.shape[0] > 0
    if have_val_loss:
        val_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Y_val, dtype=torch.float32))
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    else:
        val_loader = None

    model = MultiTaskCBEncoder(in_channels=Din, horizon=H_PRE).to(DEVICE)

    for p in model.encoder.embed_proj.parameters():
        p.requires_grad = True
    for p in model.encoder.post_proj.parameters():
        p.requires_grad = True
    for p in model.recon_head.parameters():
        p.requires_grad = True
    for p in model.forecast_head.parameters():
        p.requires_grad = True
    for p in model.proj_head.parameters():
        p.requires_grad = True

    model.encoder.print_trainable()

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=LR, weight_decay=WEIGHT_DECAY)

    df_truth_val = _truth_wide_for_val(ds_train, aux, H_PRE, train_last_day=TRAIN_LAST_DAY)

    best = {"loss": float("inf"), "overall": float("inf"), "calm": float("inf"), "storm": float("inf"), "epoch": -1}

    def compute_val_loss():
        if val_loader is None:
            return float("nan")
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)  # (B, T, 1, F)
                yb = yb.to(DEVICE)  # (B, H, 1, F)
                Bsz, T, _, _ = xb.shape

                mask_ssl = make_temporal_mask(
                    Bsz, T, mode=MASK_MODE, ratio=MASK_RATIO, span_len=SPAN_LEN, device=xb.device
                )
                xmasked = apply_mask(xb, mask_ssl)
                recon, fut = model(xmasked)
                l_ssl  = masked_recon_loss(recon, xb, mask_ssl)
                l_fore = forecast_loss(fut, yb)

                if USE_CONTRASTIVE:
                    x1 = make_view(
                        xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                        jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB
                    )
                    x2 = make_view(
                        xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                        jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB
                    )
                    z1 = model.pooled_projection(x1)
                    z2 = model.pooled_projection(x2)
                    l_contrast = info_nce_loss(z1, z2, tau=TAU)
                else:
                    l_contrast = torch.tensor(0.0, device=xb.device)

                l = W_SSL * l_ssl + W_FORECAST * l_fore + W_CONTRAST * l_contrast
                total += l.item() * xb.size(0)
                count += xb.size(0)
        return total / max(1, count)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        seen_samples = 0
        start = time.time()

        for xb, yb in tqdm(loader, desc=f"Pretrain Epoch {epoch}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)      # (B, T, 1, F)
            yb = yb.to(DEVICE)      # (B, H, 1, F)

            Bsz, T, _, _ = xb.shape
            mask_ssl = make_temporal_mask(
                Bsz, T, mode=MASK_MODE, ratio=MASK_RATIO, span_len=SPAN_LEN, device=xb.device
            )
            xmasked = apply_mask(xb, mask_ssl)

            opt.zero_grad()
            recon, fut = model(xmasked)
            loss_ssl  = masked_recon_loss(recon, xb, mask_ssl)
            loss_fore = forecast_loss(fut, yb)

            if USE_CONTRASTIVE:
                x1 = make_view(
                    xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                    jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB
                )
                x2 = make_view(
                    xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                    jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB
                )
                z1 = model.pooled_projection(x1)
                z2 = model.pooled_projection(x2)
                loss_contrast = info_nce_loss(z1, z2, tau=TAU)
            else:
                loss_contrast = torch.tensor(0.0, device=xb.device)

            loss = W_SSL * loss_ssl + W_FORECAST * loss_fore + W_CONTRAST * loss_contrast

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            epoch_loss += loss.item() * Bsz
            seen_samples += Bsz

        epoch_loss /= max(1, seen_samples)
        took = time.time() - start
        print(f"[Pretrain] Epoch {epoch:02d} total={epoch_loss:.6f} "
              f"ssl={loss_ssl.item():.6f} fore={loss_fore.item():.6f} "
              f"contrast={loss_contrast.item():.6f} time={took:.2f}s | "
              f"batches={len(loader)} samples={seen_samples}")

        # ---------- Validation: loss ----------
        val_loss = compute_val_loss()

        # ---------- Validation: rolling RMSE + calm/storm ----------
        if df_truth_val is not None and not df_truth_val.empty:
            df_pred_val = _predict_val_roll(ds_train, model, scalers, H_PRE, aux, TRAIN_LAST_DAY)
            if not df_pred_val.empty:
                (rmse_overall, zero_overall,
                 rmse_calm, zero_calm, n_calm,
                 rmse_storm, zero_storm, n_storm) = _event_rmse_blocks(
                    ds_train, df_truth_val, df_pred_val,
                    train_last_day=TRAIN_LAST_DAY,
                    method=EVENT_METHOD,
                    storm_top_q=STORM_TOP_Q,
                    calm_bottom_q=CALM_BOTTOM_Q,
                    calm_n=CALM_N,
                    seed=42
                )
            else:
                rmse_overall = rmse_calm = rmse_storm = float("nan")
                zero_overall = zero_calm = zero_storm = float("nan")
                n_calm = n_storm = 0
        else:
            rmse_overall = rmse_calm = rmse_storm = float("nan")
            zero_overall = zero_calm = zero_storm = float("nan")
            n_calm = n_storm = 0

        if np.isfinite(rmse_overall):
            print(f"[Pretrain Val] Epoch {epoch:02d}: ValLoss={val_loss:.6f} | Rolling RMSE={rmse_overall:.4f} (all-zero={zero_overall:.4f})")
        else:
            print(f"[Pretrain Val] Epoch {epoch:02d}: ValLoss={val_loss:.6f} | Rolling RMSE=nan (all-zero=nan)")

        if np.isfinite(rmse_calm):
            print(f"[EventEval] Calm-heavy ({n_calm} days): RMSE={rmse_calm:.4f} (all-zero={zero_calm:.4f})")
        else:
            print(f"[EventEval] Calm-heavy: RMSE=nan (all-zero=nan)")

        if np.isfinite(rmse_storm):
            print(f"[EventEval] Storm-heavy ({n_storm} days): RMSE={rmse_storm:.4f} (all-zero={zero_storm:.4f})")
        else:
            print(f"[EventEval] Storm-heavy: RMSE=nan (all-zero=nan)")

        # ---------- Save best by BEST_BY ----------
        metric_map = {
            "loss": val_loss,
            "overall": rmse_overall,
            "calm": rmse_calm,
            "storm": rmse_storm
        }
        key = str(BEST_BY).lower()
        chosen = metric_map.get(key, rmse_overall)

        def _is_better(new, old):
            if not np.isfinite(new):
                return False
            if not np.isfinite(old):
                return True
            return new < old

        if key == "loss":
            if _is_better(chosen, best["loss"]):
                best["loss"] = chosen
                best["epoch"] = epoch
                torch.save(
                    {
                        "cb_model_name": CB_MODEL_NAME,
                        "state_dict": model.encoder.state_dict(),
                        "d_out": D_OUT,
                        "in_channels": Din,
                        "lora_r": LORA_R,
                        "lora_alpha": LORA_ALPHA,
                        "lora_dropout": LORA_DROPOUT,
                        "val_loss": val_loss,
                        "val_rmse_overall": rmse_overall,
                        "val_rmse_calm": rmse_calm,
                        "val_rmse_storm": rmse_storm,
                        "best_by": key,
                        "epoch": epoch,
                    },
                    SAVE_PATH,
                )
                print(f"[Pretrain Val] New best (loss={val_loss:.6f}) at epoch {epoch}. Saved: {SAVE_PATH}")
        else:
            if _is_better(chosen, best[key]):
                best[key] = chosen
                best["epoch"] = epoch
                torch.save(
                    {
                        "cb_model_name": CB_MODEL_NAME,
                        "state_dict": model.encoder.state_dict(),
                        "d_out": D_OUT,
                        "in_channels": Din,
                        "lora_r": LORA_R,
                        "lora_alpha": LORA_ALPHA,
                        "lora_dropout": LORA_DROPOUT,
                        "val_loss": val_loss,
                        "val_rmse_overall": rmse_overall,
                        "val_rmse_calm": rmse_calm,
                        "val_rmse_storm": rmse_storm,
                        "best_by": key,
                        "epoch": epoch,
                    },
                    SAVE_PATH,
                )
                print(f"[Pretrain Val] New best ({key} RMSE={chosen:.6f}) at epoch {epoch}. Saved: {SAVE_PATH}")

        try:
            os.makedirs(SAVE_ADAPTER_DIR, exist_ok=True)
            model.encoder.backbone.save_pretrained(SAVE_ADAPTER_DIR)
            print(f"[Pretrain Val] Saved LoRA adapter/backbone to: {SAVE_ADAPTER_DIR}")
        except Exception as e:
            print(f"[Pretrain Val] Adapter export skipped ({e}).")

# =========================
# Main
# =========================
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    train_multitask_contrast(ds_train)
    ds_train.close()

if __name__ == "__main__":
    main()
