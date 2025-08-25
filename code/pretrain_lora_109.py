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
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

# =========================
# Config
# =========================
SPLIT_DIR   = "data/"
OUT_DIR     = "results_lora_109/"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# Windows / data
SEQ_LEN     = 24     # context length T (match downstream)
H_PRE       = 24     # pretraining forecast horizon H
BATCH_SIZE  = 1024   # batch size for training

# Training
EPOCHS          = 100
LR              = 1e-4
WEIGHT_DECAY    = 1e-3
GRAD_CLIP_NORM  = 5.0

# ---------- Masking for SSL ----------
MASK_MODE               = "span"   # "random" or "span"
MASK_RATIO              = 0.30     # fraction of timesteps masked per window
SPAN_LEN                = 4        # used if MASK_MODE=="span": contiguous length per span

# For contrastive views (lighter masking by default)
VIEW_MASK_MODE          = "span"   # "random" or "span"
VIEW_MASK_RATIO         = 0.15
VIEW_SPAN_LEN           = 3

# ---------- Contrastive (InfoNCE) ----------
USE_CONTRASTIVE         = False  #True
W_CONTRAST              = 1.0      # weight of contrastive loss
TAU                     = 0.2      # temperature for InfoNCE
ADD_JITTER_STD          = 0.01     # small Gaussian jitter std for views
ADD_DROPOUT_PROB        = 0.0      # channel dropout prob for views
# If you want contrastive gradients to update the encoder, set to True
CONTRAST_UPDATES_ENCODER = False

# ---------- Frequency-domain augmentations (TimesNet-style) ----------
USE_FREQ_AUG            = False #True
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
W_SSL       = 1.0   # masked reconstruction loss weight
W_FORECAST  = 1.0   # next-H forecasting loss weight

# ---------- NEW: Outage classification (weather-only) ----------
W_CLS             = 1.0        # weight for classification loss
CLS_POS_WEIGHT    = 6.0        # BCE pos_weight to fight imbalance (tune)
CLS_DROPOUT       = 0.1
CLS_THRESHOLD     = 0.5        # for reporting ratios at validation

# ---------- CB + PEFT (LoRA) ----------
CB_MODEL_NAME   = "/home2/caojia/weather_prediction/code/chronos-bolt-small"
D_OUT           = 64
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

# ---------- Train/Val split by days ----------
TRAIN_LAST_DAY = 80  # use days 1..80 for training; 81..end for day-ahead validation

# ---------- Storm-weighted sampling (use outage labels) ----------
STORM_SAMPLING         = True
STORM_POW              = 0.75
STORM_MIN_WEIGHT       = 0.1

# ---------- Steps-per-epoch expansion ----------
EPOCH_EXPANSION = 1 #2.0

# Save path (wrapper state_dict)
SAVE_PATH = os.path.join(OUT_DIR, f"pretrained_cb_encoder_peft_independent_H{H_PRE}.pt")
# Optional: export LoRA adapter/backbone in HF format
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
# Data prep (independent locations) — WEATHER ONLY INPUTS
# =========================
def prepare_multitask_windows_split(ds_train, seq_len=24, horizon=24, train_last_day=80):
    """
    Build training windows for **independent locations** using ONLY WEATHER FEATURES as inputs:
      X_tr:       (nW*N, seq_len, 1, F)       # weather context
      Y_fut_tr:   (nW*N, horizon, 1, F)       # weather future (forecast target)
      Y_cls_tr:   (nW*N, horizon, 1)          # outage labels (>0) for next H (classification target)
    Fit weather scalers on training portion only (days <= train_last_day).
    """
    # y: outages (for labels only), w: weather (109 vars)
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, uniq_days, day_end_idx = _compute_day_indices(ts)

    T, N, F = w.shape
    Din = F  # WEATHER ONLY

    # train-only scalers for weather
    scalers = _fit_scalers_train_only(y, w, days, train_last_day)  # weather stats are in w_mu/w_sd
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F) # (T, N, F)
    X_all = w_sc  # inputs are ONLY weather

    nW_total = T - seq_len - horizon + 1
    if nW_total <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    last_train_idx = day_end_idx[train_last_day - 1]
    nW_train = max(0, last_train_idx - seq_len - horizon + 1)

    X_list, Yw_list, Ycls_list = [], [], []
    for node in range(N):
        x_loc = X_all[:, node, :]  # (T, F)
        # weather context windows
        Xw = np.stack([x_loc[i:i+seq_len] for i in range(nW_total)], axis=0)                  # (nW_total, seq_len, F)
        # weather future windows
        Yw = np.stack([X_all[i+seq_len:i+seq_len+horizon, node, :] for i in range(nW_total)], axis=0)  # (nW_total, H, F)
        # outage labels for next H (based on ORIGINAL y, unscaled)
        Ycls = np.stack([y[i+seq_len:i+seq_len+horizon, node] for i in range(nW_total)], axis=0)       # (nW_total, H)
        Ycls = (Ycls > 0.0).astype(np.float32)  # binary labels

        if nW_train > 0:
            X_list.append(Xw[:nW_train][:, :, None, :])        # (nW_train, seq_len, 1, F)
            Yw_list.append(Yw[:nW_train][:, :, None, :])       # (nW_train, H, 1, F)
            Ycls_list.append(Ycls[:nW_train][:, :, None])      # (nW_train, H, 1)

    if not X_list:
        X_tr = np.empty((0, seq_len, 1, Din), dtype=float)
        Y_fut_tr = np.empty((0, horizon, 1, Din), dtype=float)
        Y_cls_tr = np.empty((0, horizon, 1), dtype=float)
    else:
        X_tr     = np.concatenate(X_list, axis=0)
        Y_fut_tr = np.concatenate(Yw_list, axis=0)
        Y_cls_tr = np.concatenate(Ycls_list, axis=0)

    aux = {
        "X_all_scaled": X_all,     # (T, N, F)
        "timestamps": ts,          # (T,)
        "days": days,              # (T,)
        "day_end_idx": day_end_idx,# list[int]
        "y_outages": y,            # (T, N) for building validation labels
    }
    return X_tr, Y_fut_tr, Y_cls_tr, Din, scalers, aux

def build_val_windows_day_ahead(aux, horizon, seq_len, train_last_day=80):
    """
    Validation windows (day-ahead) using ONLY weather inputs.
      Returns:
        Xv      : (nV, seq_len, 1, F)
        Yw_val  : (nV, horizon, 1, F)
        Ycls_val: (nV, horizon, 1)  (binary labels from outages)
    """
    X_all = aux["X_all_scaled"]           # (T, N, F)
    y     = aux["y_outages"]              # (T, N)
    ts    = aux["timestamps"]
    days  = aux["days"]
    day_end_idx = aux["day_end_idx"]
    T, N, F = X_all.shape

    Xv, Yw, Ycls = [], [], []
    last_day = int(days.max())
    for d in range(train_last_day, last_day):
        end_idx = day_end_idx[d - 1]
        start_ctx = end_idx - (seq_len - 1)
        fut_start = end_idx + 1
        fut_end   = fut_start + horizon - 1
        if start_ctx < 0 or fut_end >= T:
            continue
        for node in range(N):
            ctx = X_all[start_ctx:end_idx+1, node, :].reshape(seq_len, 1, F)
            fut_w = X_all[fut_start:fut_start+horizon, node, :].reshape(horizon, 1, F)
            fut_y = y[fut_start:fut_start+horizon, node].reshape(horizon, 1)
            Xv.append(ctx)
            Yw.append(fut_w)
            Ycls.append((fut_y > 0.0).astype(np.float32))

    if not Xv:
        return (np.empty((0, seq_len, 1, F), dtype=float),
                np.empty((0, horizon, 1, F), dtype=float),
                np.empty((0, horizon, 1), dtype=float))
    Xv = np.stack(Xv, axis=0)
    Yw = np.stack(Yw, axis=0)
    Ycls = np.stack(Ycls, axis=0)
    return Xv, Yw, Ycls

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
    # Weather-only scalers (fit on train split)
    mask = days <= train_last_day
    w_tr = w[mask, :, :].reshape(-1, w.shape[2])
    w_mu, w_sd = zfit(w_tr)
    # keep outage stats placeholders in case of downstream code (not used here)
    y_mu, y_sd = np.array([0.0]), np.array([1.0])
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
    """
    Wrap a Chronos/Chronos-Bolt backbone with LoRA (PEFT) and simple
    input/output adapters to produce (B, T, N, D_OUT) embeddings.
    """
    def __init__(self, in_channels, cb_model_name=CB_MODEL_NAME, d_out=D_OUT,
                 lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT):
        super().__init__()
        if not HAVE_TRANSFORMERS:
            raise ImportError(
                f"transformers not available: {_transformers_err}\n"
                "Install: pip install transformers accelerate safetensors"
            )
        if not HAVE_PEFT:
            raise ImportError(
                f"peft not available: {_peft_err}\n"
                "Install: pip install peft"
            )

        self.in_channels = in_channels
        self.d_out = d_out

        self.config = AutoConfig.from_pretrained(cb_model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            cb_model_name,
            trust_remote_code=True,
            output_hidden_states=True
        )

        targets = _discover_lora_targets(self.backbone)
        print(f"[PEFT] LoRA targets: {targets}")

        lcfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        try:
            self.backbone = get_peft_model(self.backbone, lcfg)
        except ValueError as e:
            print(f"[PEFT] Warning: {e} -> fallback target_modules=['linear']")
            lcfg.target_modules = ["linear"]
            self.backbone = get_peft_model(self.backbone, lcfg)

        hidden_dim = getattr(self.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.config, "d_model", None)
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
            out = self.backbone.encoder(
                inputs_embeds=emb,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        else:
            out = self.backbone(
                inputs_embeds=emb,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

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
# Multi-task model using CB encoder (WEATHER ONLY INPUTS)
# =========================
class MultiTaskCBEncoder(nn.Module):
    """
    Encoder + heads:
      - recon_head: reconstruct inputs at masked timesteps (SSL)  -> weather (F)
      - forecast_head: predict next H steps of weather inputs     -> weather (F)
      - proj_head: projection for InfoNCE (contrastive)
      - cls_head: predict outage occurrence (>0) for next H steps from weather-only context
    """
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
        self.cls_head = nn.Sequential(
            nn.Linear(D_OUT, 2 * D_OUT),
            nn.GELU(),
            nn.Dropout(CLS_DROPOUT),
            nn.Linear(2 * D_OUT, horizon)  # logits per future step
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

    def outage_logits(self, x):
        """
        x: (B,T,N,F) weather-only context
        Returns logits: (B,H,N)
        """
        h = self.encoder(x)            # (B,T,N,D_OUT)
        pooled = h.mean(dim=1)         # (B,N,D_OUT)
        logits = self.cls_head(pooled) # (B,N,H)
        return logits.permute(0, 2, 1) # (B,H,N)

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

# ---------- TSMixup helpers ----------
def _dirichlet_weights(batch_size, k, alpha, device):
    if k == 1:
        w = torch.ones(batch_size, 1, device=device)
    else:
        w = torch.distributions.Dirichlet(alpha * torch.ones(k, device=device)).sample((batch_size,))
    return w  # (B, k)

def _gather_k_partners(x, k, device):
    B = x.size(0)
    base = torch.arange(B, device=device)
    idx = base.view(B, 1).repeat(1, k)
    if k > 1:
        partners = torch.randint(0, B, (B, k-1), device=device)
        idx[:, 1:] = partners
    x_g = x[idx.reshape(-1)].view(B, k, *x.shape[1:])
    return x_g, idx

def _mean_abs_scale(x, dims, eps):
    s = x.abs().mean(dim=dims, keepdim=True).clamp_min(eps)
    return x / s, s

def tsmixup_batch(
    xb, yb=None, p=0.5, k_max=3, alpha=1.5,
    segment=False, len_frac=(0.5, 1.0), eps=1e-6
):
    if (p <= 0.0) or (xb.size(0) < 2) or (torch.rand(1, device=xb.device).item() > p):
        return xb, yb

    B, T, N, F = xb.shape
    H = yb.size(1) if yb is not None else None

    k = int(torch.randint(1, k_max + 1, (1,), device=xb.device).item())
    if k == 1:
        return xb, yb

    xb_g, idx = _gather_k_partners(xb, k, xb.device)
    if yb is not None:
        yb_g = yb[idx.reshape(-1)].view(B, k, *yb.shape[1:])
    else:
        yb_g = None

    w = _dirichlet_weights(B, k, alpha, xb.device).view(B, k, 1, 1, 1)

    if not segment:
        xb_scaled, _ = _mean_abs_scale(xb_g, dims=(2, 3, 4), eps=eps)
        xb_mix = (w * xb_scaled).sum(dim=1)
        if yb_g is not None:
            yb_scaled, _ = _mean_abs_scale(yb_g, dims=(2, 3, 4), eps=eps)
            yb_mix = (w * yb_scaled).sum(dim=1)
        else:
            yb_mix = None
        return xb_mix, yb_mix

    f = float(torch.empty((), device=xb.device).uniform_(len_frac[0], len_frac[1]).item())
    Lx = max(1, int(round(f * T)))
    sx = 0 if T == Lx else int(torch.randint(0, T - Lx + 1, (1,), device=xb.device).item())
    if H is not None:
        Ly = max(1, int(round(f * H)))
        sy = 0 if H == Ly else int(torch.randint(0, H - Ly + 1, (1,), device=xb.device).item())

    xb_out = xb.clone()
    xb_seg = xb_g[:, :, sx:sx+Lx, :, :]
    xb_seg_scaled, _ = _mean_abs_scale(xb_seg, dims=(2, 3, 4), eps=eps)
    xb_out[:, sx:sx+Lx, :, :] = (w * xb_seg_scaled).sum(dim=1)

    if yb_g is not None:
        yb_out = yb.clone()
        yb_seg = yb_g[:, :, sy:sy+Ly, :, :]
        yb_seg_scaled, _ = _mean_abs_scale(yb_seg, dims=(2, 3, 4), eps=eps)
        yb_out[:, sy:sy+Ly, :, :] = (w * yb_seg_scaled).sum(dim=1)
    else:
        yb_out = None

    return xb_out, yb_out

# =========================
# Small classification metrics (validation)
# =========================
@torch.no_grad()
def _cls_ratios_from_logits(logits, y_true, thr=0.5):
    """
    logits: (B,H,1) or (B,H,N)   y_true: same shape, values in {0,1}
    returns: zero_to_nonzero, nonzero_to_zero
    """
    p = torch.sigmoid(logits)
    pred = (p >= thr).float()
    y = y_true.float()

    zeros = (y == 0.0)
    ones  = (y == 1.0)

    z2n = (pred * zeros).sum().item()
    nzeros = zeros.sum().item()
    zero_to_nonzero = (z2n / nzeros) if nzeros > 0 else float("nan")

    n2z = ((1.0 - pred) * ones).sum().item()
    nonzeros = ones.sum().item()
    nonzero_to_zero = (n2z / nonzeros) if nonzeros > 0 else float("nan")

    return zero_to_nonzero, nonzero_to_zero

# =========================
# Training loop
# =========================
def train_multitask(ds_train):
    # --- Build split-aware train windows (WEATHER ONLY inputs) ---
    X_tr, Yw_fut_tr, Ycls_tr, Din, scalers, aux = prepare_multitask_windows_split(
        ds_train, seq_len=SEQ_LEN, horizon=H_PRE, train_last_day=TRAIN_LAST_DAY
    )
    dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),        # (B,T,1,F)
        torch.tensor(Yw_fut_tr, dtype=torch.float32),   # (B,H,1,F)
        torch.tensor(Ycls_tr, dtype=torch.float32),     # (B,H,1)
    )

    # Storm-aware sample weights based on outage labels prevalence in H
    if STORM_SAMPLING and len(dataset) > 0:
        with torch.no_grad():
            y_cls = torch.tensor(Ycls_tr, dtype=torch.float32)  # (nS, H, 1)
            s = y_cls.mean(dim=(1, 2)).clamp_min(0) + 1e-6
            w = (s ** STORM_POW)
            w = (w / (w.mean() + 1e-6)).clamp_min(STORM_MIN_WEIGHT)
            sample_weights = w.numpy()
    else:
        sample_weights = None

    # Sampler
    if EPOCH_EXPANSION and EPOCH_EXPANSION > 1.0:
        num_samples = int(len(dataset) * float(EPOCH_EXPANSION))
        if STORM_SAMPLING and sample_weights is not None and len(sample_weights) > 0:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
        else:
            sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        if STORM_SAMPLING and sample_weights is not None and len(sample_weights) > 0:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Build validation windows (day-ahead) for losses/ratios ---
    X_val, Yw_val, Ycls_val = build_val_windows_day_ahead(aux, horizon=H_PRE, seq_len=SEQ_LEN, train_last_day=TRAIN_LAST_DAY)
    have_val = X_val.shape[0] > 0
    if have_val:
        val_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Yw_val, dtype=torch.float32),
                                torch.tensor(Ycls_val, dtype=torch.float32))
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    else:
        val_loader = None

    # --- Model ---
    model = MultiTaskCBEncoder(in_channels=Din, horizon=H_PRE).to(DEVICE)

    # Ensure trainable parts include LoRA + our adapters & heads
    for p in model.encoder.embed_proj.parameters(): p.requires_grad = True
    for p in model.encoder.post_proj.parameters():  p.requires_grad = True
    for p in model.recon_head.parameters():         p.requires_grad = True
    for p in model.forecast_head.parameters():      p.requires_grad = True
    for p in model.proj_head.parameters():          p.requires_grad = True
    for p in model.cls_head.parameters():           p.requires_grad = True

    model.encoder.print_trainable()

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=LR, weight_decay=WEIGHT_DECAY)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CLS_POS_WEIGHT], device=DEVICE))

    def compute_val_metrics():
        if val_loader is None:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        model.eval()
        total = 0.0
        n = 0
        loss_ssl_sum = 0.0
        loss_fore_sum = 0.0
        loss_cls_sum = 0.0
        z2n_list, n2z_list = [], []
        with torch.no_grad():
            for xb, yw, ycls in val_loader:
                xb = xb.to(DEVICE)      # (B,T,1,F) weather
                yw = yw.to(DEVICE)      # (B,H,1,F)
                ycls = ycls.to(DEVICE)  # (B,H,1) binary

                Bsz, T, _, Fdim = xb.shape
                mask_ssl = make_temporal_mask(
                    Bsz, T, mode=MASK_MODE, ratio=MASK_RATIO, span_len=SPAN_LEN, device=xb.device
                )
                xmasked = apply_mask(xb, mask_ssl)

                recon, fut = model(xmasked)  # recon: (B,T,1,F)  fut: (B,H,1,F)
                logits = model.outage_logits(xb)  # (B,H,1)

                l_ssl  = masked_recon_loss(recon, xb, mask_ssl)
                l_fore = forecast_loss(fut, yw)
                l_cls  = bce(logits.squeeze(-1), ycls.squeeze(-1))

                loss_ssl_sum += l_ssl.item() * xb.size(0)
                loss_fore_sum += l_fore.item() * xb.size(0)
                loss_cls_sum  += l_cls.item()  * xb.size(0)

                z2n, n2z = _cls_ratios_from_logits(logits, ycls, thr=CLS_THRESHOLD)
                z2n_list.append(z2n)
                n2z_list.append(n2z)

                total += (W_SSL*l_ssl + W_FORECAST*l_fore + W_CLS*l_cls).item() * xb.size(0)
                n += xb.size(0)
        return (total/max(1,n),
                loss_ssl_sum/max(1,n),
                loss_fore_sum/max(1,n),
                loss_cls_sum/max(1,n),
                float(np.nanmean(z2n_list)) if z2n_list else float("nan"),
                float(np.nanmean(n2z_list)) if n2z_list else float("nan"))

    best = {"loss": float("inf"), "epoch": -1}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        seen_samples = 0
        start = time.time()

        for xb, yw, ycls in tqdm(loader, desc=f"Pretrain Epoch {epoch}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)      # (B, T, 1, F) weather
            yw = yw.to(DEVICE)      # (B, H, 1, F) weather future
            ycls = ycls.to(DEVICE)  # (B, H, 1) outage labels (binary)

            # TSMixup for weather tasks (we do NOT mix classification labels to keep them crisp)
            xb, yw = tsmixup_batch(
                xb, yw,
                p=0.5,
                k_max=3,
                alpha=1.5,
                segment=False,
                len_frac=(0.5, 1.0),
                eps=1e-6
            )

            Bsz, T, _, _ = xb.shape
            mask_ssl = make_temporal_mask(
                Bsz, T, mode=MASK_MODE, ratio=MASK_RATIO, span_len=SPAN_LEN, device=xb.device
            )
            xmasked = apply_mask(xb, mask_ssl)

            opt.zero_grad()
            recon, fut = model(xmasked)  # recon: (B,T,1,F)  fut: (B,H,1,F)

            loss_ssl  = masked_recon_loss(recon, xb, mask_ssl)
            loss_fore = forecast_loss(fut, yw)

            # classification from WEATHER ONLY context (unmasked xb)
            logits = model.outage_logits(xb).squeeze(-1)   # (B,H)
            loss_cls = bce(logits, ycls.squeeze(-1))       # BCE with logits

            # Contrastive (optional)
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

            loss = W_SSL * loss_ssl + W_FORECAST * loss_fore + W_CLS * loss_cls + W_CONTRAST * loss_contrast
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            epoch_loss += loss.item() * Bsz
            seen_samples += Bsz

        epoch_loss /= max(1, seen_samples)
        took = time.time() - start
        print(f"[Pretrain] Epoch {epoch:02d} total={epoch_loss:.6f} "
              f"ssl={loss_ssl.item():.6f} fore={loss_fore.item():.6f} "
              f"cls={loss_cls.item():.6f} contrast={loss_contrast.item():.6f} "
              f"time={took:.2f}s | batches={len(loader)} samples={seen_samples}")

        # ---------- Validation: losses & classification ratios ----------
        (val_total, val_ssl, val_fore, val_cls, z2n_ratio, n2z_ratio) = compute_val_metrics()
        print(f"[Pretrain Val] Epoch {epoch:02d}: "
              f"Total={val_total:.6f} | Recon={val_ssl:.6f} | Forecast={val_fore:.6f} | Classif={val_cls:.6f} "
              f"| Zero→Nonzero={z2n_ratio:.4f} | Nonzero→Zero={n2z_ratio:.4f}")

        # ---------- Save best by total loss ----------
        def _is_better(new, old):
            if not np.isfinite(new): return False
            if not np.isfinite(old): return True
            return new < old

        if _is_better(val_total, best["loss"]):
            best["loss"] = val_total
            best["epoch"] = epoch

            ckpt_payload = {
                "cb_model_name": CB_MODEL_NAME,
                "state_dict": model.encoder.state_dict(),   # encoder weights
                "d_out": D_OUT,
                "in_channels": Din,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                # keep useful monitors
                "val_total": val_total,
                "val_recon": val_ssl,
                "val_forecast": val_fore,
                "val_classif": val_cls,
                "epoch": epoch,
                # ------- save classifier head so finetuning can gate with it -------
                "cls_head_state_dict": model.cls_head.state_dict(),
            }
            torch.save(ckpt_payload, SAVE_PATH)
            print(f"[Pretrain Val] New best (total loss={val_total:.6f}) at epoch {epoch}. Saved: {SAVE_PATH}")

            # Optional: save LoRA adapter/backbone
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
    train_multitask(ds_train)
    ds_train.close()

if __name__ == "__main__":
    main()
