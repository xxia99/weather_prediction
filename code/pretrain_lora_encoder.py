import os
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
from moe import MoE


class FocalLoss(nn.Module):
    """Binary focal loss with optional class balancing."""
    def __init__(self, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                                 pos_weight=torch.tensor([self.pos_weight], device=logits.device))
        prob = torch.sigmoid(logits)
        pt = prob * target + (1 - prob) * (1 - target)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()

# =========================
# Config
# =========================
SPLIT_DIR   = "data/"
OUT_DIR     = "results_lora/new"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

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
W_SSL       = 1.0   # masked reconstruction loss weight (weather-only)
W_FORECAST  = 1.0   # next-H forecasting loss weight (weather-only)
# Extra emphasis on outage channel (index 0) -- NOT USED now since we don't reconstruct/forecast outage
W_OUTAGE_SSL      = 0#0.5
W_OUTAGE_FORECAST = 0#1.0

# ---------- NEW: Outage classification (weather-only) ----------
W_CLS             = 1.0        # weight for classification loss
CLS_POS_WEIGHT    = 6.0        # BCE/Focal pos_weight to fight imbalance (tune)
CLS_DROPOUT       = 0.1
FOCAL_GAMMA       = 2.0        # focusing parameter for FocalLoss
CURRICULUM_EPOCHS = 10         # warmup epochs for classification loss
W_MAG             = 1.0        # weight for outage magnitude loss
W_SPIKE           = 0.1        # weight for spike quantile loss
QUANTILE_TAU      = 0.9        # high quantile for residual spikes
NOISE_AUG_PROB    = 0.3        # probability of noise perturbation
NOISE_AUG_STD     = 0.01       # std of Gaussian noise for augmentation

# ---------- Mixture-of-Experts ----------
MOE_EXPERTS = 4

# ---------- CB + PEFT (LoRA) ----------
CB_MODEL_NAME   = "/home2/caojia/weather_prediction/code/chronos-bolt-small"
D_OUT           = 64
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

# ---------- Train/Val split by days ----------
TRAIN_LAST_DAY = 80  # use days 1..80 for training; 81..end for day-ahead validation

# ---------- Event-stratified validation (kept but unused now) ----------
EVENT_METHOD   = "mean"   # or "max" for day score
STORM_TOP_Q    = 0.80     # top 20% days considered "stormy"
CALM_BOTTOM_Q  = 0.50     # bottom 50% pool for calm sampling
CALM_N         = 10       # sample 10 calm days
BEST_BY        = "loss"  # "overall" | "calm" | "storm" | "loss"

# ---------- TSMixup ----------
TSMIXUP_PROB       = 0#0.5
TSMIXUP_K_MAX      = 3
TSMIXUP_ALPHA      = 1.5
TSMIXUP_SEGMENT    = False
TSMIXUP_LEN_FRAC   = (0.5, 1.0)
TSMIXUP_EPS        = 1e-6

# ---------- Storm-weighted sampling ----------
STORM_SAMPLING         = True
STORM_POW              = 0.75
STORM_MIN_WEIGHT       = 0.1

# --- Calm oversampling & smoothing ---
CALM_OVERSAMPLE = 2.0
OUTAGE_THRESHOLD = 0.5
SMOOTH_WINDOW = 3

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

def channel_mse(pred, target, mask=None, ch=0):
    pred_ch = pred[..., ch:ch+1]
    targ_ch = target[..., ch:ch+1]
    if mask is None:
        return ((pred_ch - targ_ch) ** 2).mean()
    m = mask.expand_as(pred_ch)
    denom = m.sum().clamp_min(1.0)
    return ((pred_ch - targ_ch) ** 2 * m).sum() / denom

# =========================
# Data prep (independent locations)
# =========================
def prepare_multitask_windows_split(ds_train, seq_len=24, horizon=24, train_last_day=80):
    """
    Build training windows for **independent locations**:
      X_tr: (nW*N, seq_len, 1, Din=110)
      Y_tr: (nW*N, horizon, 1, Din=110)
    Fit scalers on training portion only (days <= train_last_day).
    """
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F=109)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, uniq_days, day_end_idx = _compute_day_indices(ts)

    T, N, F = w.shape
    Din = 1 + F  # keep input at 110 (1 outage + 109 weather)

    # train-only scalers
    scalers = _fit_scalers_train_only(y, w, days, train_last_day)
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)    # (T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F) # (T, N, 109)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)                             # (T, N, 110)

    nW_total = T - seq_len - horizon + 1
    if nW_total <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    last_train_idx = day_end_idx[train_last_day - 1]
    nW_train = max(0, last_train_idx - seq_len - horizon + 1)

    X_list, Y_list = [], []
    for node in range(N):
        x_loc = X_all[:, node, :]  # (T, 110)
        Xw = np.stack([x_loc[i:i+seq_len] for i in range(nW_total)], axis=0)                  # (nW_total, seq_len, 110)
        Yw = np.stack([X_all[i+seq_len:i+seq_len+horizon, node, :] for i in range(nW_total)], axis=0)  # (nW_total, H, 110)
        if nW_train > 0:
            X_list.append(Xw[:nW_train][:, :, None, :])   # (nW_train, seq_len, 1, 110)
            Y_list.append(Yw[:nW_train][:, :, None, :])   # (nW_train, H, 1, 110)

    if not X_list:
        X_tr = np.empty((0, seq_len, 1, Din), dtype=float)
        Y_tr = np.empty((0, horizon, 1, Din), dtype=float)
    else:
        X_tr = np.concatenate(X_list, axis=0)
        Y_tr = np.concatenate(Y_list, axis=0)

    aux = {
        "X_all_scaled": X_all,     # (T, N, 110)
        "y_scaled": y_sc,          # (T, N)
        "timestamps": ts,          # (T,)
        "days": days,              # (T,)
        "day_end_idx": day_end_idx # list[int]
    }
    return X_tr, Y_tr, Din, scalers, aux

def build_val_windows_day_ahead(aux, horizon, seq_len, train_last_day=80):
    X_all = aux["X_all_scaled"]           # (T, N, 110)
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
# Multi-task model using CB encoder
# =========================
class MultiTaskCBEncoder(nn.Module):
    """
    Encoder + heads:
      - recon_head: reconstruct **weather-only (109)** at masked timesteps
      - forecast_head: predict next H steps of **weather-only (109)**
      - proj_head: projection for InfoNCE (contrastive, optional)
      - cls_head: predict outage occurrence (>0) for next H steps, using WEATHER-ONLY context
    """
    def __init__(self, in_channels, horizon):
        super().__init__()
        self.horizon = horizon
        self.in_channels = in_channels            # 110
        self.weather_dim = in_channels - 1        # 109
        self.encoder = ChronosBoltEncoderPEFT(in_channels=in_channels, d_out=D_OUT)

        # weather-only heads
        self.recon_head = nn.Linear(D_OUT, self.weather_dim)  # 109
        self.forecast_head = MoE(
            D_OUT,
            horizon * self.weather_dim,
            num_experts=MOE_EXPERTS,
            hidden_dim=2 * D_OUT,
            dropout=0.2,
        )
        self.proj_head = nn.Sequential(
            nn.Linear(D_OUT, D_OUT),
            nn.GELU(),
            nn.Linear(D_OUT, D_OUT)
        )
        # binary classifier and magnitude heads for outages
        self.cls_head = MoE(
            D_OUT,
            horizon,
            num_experts=MOE_EXPERTS,
            hidden_dim=2 * D_OUT,
            dropout=CLS_DROPOUT,
        )
        self.rate_head = MoE(
            D_OUT,
            horizon,
            num_experts=MOE_EXPERTS,
            hidden_dim=2 * D_OUT,
            dropout=CLS_DROPOUT,
        )
        self.spike_head = nn.Linear(D_OUT, horizon)

    def encode_steps(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)                    # (B, T, N, D_OUT)
        recon = self.recon_head(h)             # (B, T, N, 109)
        pooled = h.mean(dim=1)                 # (B, N, D_OUT)
        fut = self.forecast_head(pooled)       # (B, N, H*109)
        B, N, _ = fut.shape
        fut = fut.view(B, N, self.horizon, self.weather_dim).permute(0, 2, 1, 3)  # (B, H, N, 109)
        return recon, fut

    def pooled_projection(self, x):
        h = self.encoder(x) if CONTRAST_UPDATES_ENCODER else self.encoder(x).detach()
        pooled = h.mean(dim=1).mean(dim=1)       # (B, D_OUT)
        z = self.proj_head(pooled)               # (B, D_OUT)
        z = F.normalize(z, dim=-1)
        return z

    # ------- produce outage logits from WEATHER-ONLY context -------
    def outage_logits_from_weather(self, x):
        """
        x: (B,T,N,110) scaled inputs (outage channel is index 0).
        We zero the outage channel -> weather-only.
        Returns: logits (B, H, N)
        """
        x_wo = x.clone()
        x_wo[..., 0] = 0.0  # remove outage info from context
        h = self.encoder(x_wo)            # (B,T,N,D_OUT)
        pooled = h.mean(dim=1)            # (B,N,D_OUT)
        logits = self.cls_head(pooled)    # (B,N,H)
        return logits.permute(0, 2, 1)    # (B,H,N)

    def outage_rate_from_weather(self, x):
        x_wo = x.clone()
        x_wo[..., 0] = 0.0
        h = self.encoder(x_wo)
        pooled = h.mean(dim=1)
        rate = F.softplus(self.rate_head(pooled))
        return rate.permute(0, 2, 1)

    def outage_spike_from_weather(self, x):
        x_wo = x.clone()
        x_wo[..., 0] = 0.0
        h = self.encoder(x_wo)
        pooled = h.mean(dim=1)
        spike = self.spike_head(pooled)
        return F.relu(spike).permute(0, 2, 1)

# =========================
# Losses
# =========================
def masked_recon_loss(pred, target, mask):
    """
    pred:   (B,T,N,109)  weather-only reconstruction
    target: (B,T,N,109)
    mask:   (B,T,1,1)    time mask
    """
    m = mask.expand_as(pred)
    denom = m.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * m).sum() / denom

def forecast_loss(pred, target):
    """
    pred:   (B,H,N,109)  weather-only forecast
    target: (B,H,N,109)
    """
    return torch.mean((pred - target) ** 2)

def quantile_loss(pred, target, tau):
    diff = target - pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1) * diff))

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
# Small eval helpers (kept for compatibility; RMSE not used here)
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

# ------- zero/nonzero ratio helper (for classifier) -------
def _zero_nonzero_ratios_from_logits(logits, y_true_bin, thr=0.5):
    """
    logits: (num_batches, H) or (B,H,N) flattened -> probabilities for non-zero
    y_true_bin: same shape, 0/1 ground truth
    Returns Zero→Nonzero and Nonzero→Zero.
    """
    p = torch.sigmoid(logits)
    pred = (p >= thr).to(torch.int)
    true = y_true_bin.to(torch.int)

    true_zero = (true == 0).sum().item()
    true_nonzero = (true == 1).sum().item()
    z2n = ((true == 0) & (pred == 1)).sum().item() / true_zero if true_zero > 0 else float("nan")
    n2z = ((true == 1) & (pred == 0)).sum().item() / true_nonzero if true_nonzero > 0 else float("nan")

    # All-zero baseline: predict 0 for all
    z2n0 = 0.0 if true_zero > 0 else float("nan")
    n2z0 = 1.0 if true_nonzero > 0 else float("nan")
    return float(z2n), float(n2z), float(z2n0), float(n2z0)

def _per_day_outage_score(ds_train, method: str = "mean"):
    ts = ds_train.timestamp.values
    days, _, _ = _compute_day_indices(ts)
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)  # (T, N)
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

# =========================
# Training loop (validate losses + classifier ratios)
# =========================
def train_multitask_contrast(ds_train):
    # --- Build split-aware train windows (fit scalers on days 1..80) ---
    X_tr, Y_fut_tr, Din, scalers, aux = prepare_multitask_windows_split(
        ds_train, seq_len=SEQ_LEN, horizon=H_PRE, train_last_day=TRAIN_LAST_DAY
    )
    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                            torch.tensor(Y_fut_tr, dtype=torch.float32))
    # derive positive outage ratio for FocalLoss
    y_future_out = Y_fut_tr[..., 0]
    pos_ratio = float((y_future_out > 0).mean())
    if pos_ratio > 0 and pos_ratio < 1:
        cls_pos_weight = (1 - pos_ratio) / pos_ratio
    else:
        cls_pos_weight = CLS_POS_WEIGHT
    print(f"[Pretrain] outage positive ratio={pos_ratio:.4f} -> pos_weight={cls_pos_weight:.2f}")

    sample_weights = np.ones(len(dataset), dtype=float)
    if STORM_SAMPLING and len(dataset) > 0:
        with torch.no_grad():
            y_h = torch.tensor(Y_fut_tr[..., 0], dtype=torch.float32)  # (nS, H, 1)
            s = y_h.mean(dim=(1, 2)).clamp_min(0) + 1e-6
            w = (s ** STORM_POW)
            w = (w / (w.mean() + 1e-6)).clamp_min(STORM_MIN_WEIGHT)
            sample_weights *= w.numpy()
    if CALM_OVERSAMPLE > 1.0 and len(dataset) > 0:
        with torch.no_grad():
            y_h = torch.tensor(Y_fut_tr[..., 0], dtype=torch.float32)
            calm_mask = (y_h.mean(dim=(1, 2)) <= OUTAGE_THRESHOLD).numpy()
            sample_weights *= np.where(calm_mask, CALM_OVERSAMPLE, 1.0)
    if np.allclose(sample_weights, 1.0):
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

    # --- Build validation windows (day-ahead) for loss/ratios ---
    X_val, Y_val = build_val_windows_day_ahead(aux, horizon=H_PRE, seq_len=SEQ_LEN, train_last_day=TRAIN_LAST_DAY)
    have_val = X_val.shape[0] > 0
    if have_val:
        val_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Y_val, dtype=torch.float32))
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Focal loss with class imbalance handling
    bce = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=cls_pos_weight)

    best = {"loss": float("inf"), "epoch": -1}

    def compute_val_metrics():
        if val_loader is None:
            return (float("nan"), float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        model.eval()
        tot, cnt = 0.0, 0
        ssl_sum, fore_sum, cls_sum, mag_sum, spike_sum = 0.0, 0.0, 0.0, 0.0, 0.0
        # for ratios
        logits_all = []
        ycls_all   = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)  # (B, T, 1, 110)
                yb = yb.to(DEVICE)  # (B, H, 1, 110)

                Bsz, T, _, _ = xb.shape
                mask_ssl = make_temporal_mask(
                    Bsz, T, mode=MASK_MODE, ratio=MASK_RATIO, span_len=SPAN_LEN, device=xb.device
                )
                xmasked = apply_mask(xb, mask_ssl)

                recon, fut = model(xmasked)  # recon: (B,T,1,109)  fut: (B,H,1,109)

                # weather-only targets
                xb_weather = xb[..., 1:]       # (B,T,1,109)
                yb_weather = yb[..., 1:]       # (B,H,1,109)

                l_ssl  = masked_recon_loss(recon, xb_weather, mask_ssl)
                l_fore = forecast_loss(fut, yb_weather)

                y_future_out = yb[..., 0].squeeze(-1)
                y_cls = (y_future_out > 0).float()
                logits = model.outage_logits_from_weather(xb).squeeze(-1)
                rate = model.outage_rate_from_weather(xb).squeeze(-1)
                spike = model.outage_spike_from_weather(xb).squeeze(-1)
                l_cls = bce(logits, y_cls)
                lam = torch.sigmoid(logits) * (rate + spike)
                l_mag = F.poisson_nll_loss(lam + 1e-8, y_future_out, log_input=False)
                pos_mask = y_cls.bool()
                if pos_mask.any():
                    residual = F.relu(y_future_out[pos_mask] - rate[pos_mask])
                    l_spike = quantile_loss(spike[pos_mask], residual, QUANTILE_TAU)
                else:
                    l_spike = torch.tensor(0.0, device=xb.device)

                ssl_sum  += l_ssl.item()  * Bsz
                fore_sum += l_fore.item() * Bsz
                cls_sum  += l_cls.item()  * Bsz
                mag_sum  += l_mag.item()  * Bsz
                spike_sum += l_spike.item() * Bsz

                if USE_CONTRASTIVE:
                    x1 = make_view(xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                                   jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB)
                    x2 = make_view(xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                                   jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB)
                    z1 = model.pooled_projection(x1)
                    z2 = model.pooled_projection(x2)
                    l_contrast = info_nce_loss(z1, z2, tau=TAU)
                else:
                    l_contrast = torch.tensor(0.0, device=xb.device)

                l_total = (W_SSL * l_ssl + W_FORECAST * l_fore +
                           W_CONTRAST * l_contrast + W_CLS * l_cls +
                           W_MAG * l_mag + W_SPIKE * l_spike)

                tot  += l_total.item() * Bsz
                cnt  += Bsz

                # collect for ratios
                logits_all.append(logits.detach().cpu())
                ycls_all.append(y_cls.detach().cpu())

        # aggregate ratios on CPU
        logits_cat = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
        ycls_cat   = torch.cat(ycls_all, dim=0)   if ycls_all   else torch.empty(0)
        if logits_cat.numel() > 0 and ycls_cat.numel() > 0:
            z2n, n2z, z2n0, n2z0 = _zero_nonzero_ratios_from_logits(logits_cat, ycls_cat, thr=0.5)
        else:
            z2n = n2z = z2n0 = n2z0 = float("nan")

        return (tot / max(1, cnt),
                ssl_sum / max(1, cnt),
                fore_sum / max(1, cnt),
                cls_sum / max(1, cnt),
                mag_sum / max(1, cnt),
                spike_sum / max(1, cnt),
                z2n, n2z, z2n0, n2z0)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        seen_samples = 0
        start = time.time()
        cls_scale = min(1.0, epoch / float(CURRICULUM_EPOCHS))

        for xb, yb in tqdm(loader, desc=f"Pretrain Epoch {epoch}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)      # (B, T, 1, 110)
            yb = yb.to(DEVICE)      # (B, H, 1, 110)

            # TSMixup (optional)
            xb, yb = tsmixup_batch(
                xb, yb,
                p=TSMIXUP_PROB,
                k_max=TSMIXUP_K_MAX,
                alpha=TSMIXUP_ALPHA,
                segment=TSMIXUP_SEGMENT,
                len_frac=TSMIXUP_LEN_FRAC,
                eps=TSMIXUP_EPS
            )

            if NOISE_AUG_PROB > 0.0:
                neg_mask = (yb[..., 0].max(dim=1)[0].squeeze(-1) <= OUTAGE_THRESHOLD)
                rand_mask = (torch.rand(neg_mask.shape[0], device=xb.device) < NOISE_AUG_PROB) & neg_mask
                if rand_mask.any():
                    xb[rand_mask] = xb[rand_mask] + NOISE_AUG_STD * torch.randn_like(xb[rand_mask])

            Bsz, T, _, _ = xb.shape
            mask_ssl = make_temporal_mask(
                Bsz, T, mode=MASK_MODE, ratio=MASK_RATIO, span_len=SPAN_LEN, device=xb.device
            )
            xmasked = apply_mask(xb, mask_ssl)

            opt.zero_grad()
            recon, fut = model(xmasked)  # recon: (B,T,1,109), fut: (B,H,1,109)

            # weather-only targets
            xb_weather = xb[..., 1:]       # (B,T,1,109)
            yb_weather = yb[..., 1:]       # (B,H,1,109)

            loss_ssl  = masked_recon_loss(recon, xb_weather, mask_ssl)
            loss_fore = forecast_loss(fut, yb_weather)

            # classification + magnitude
            y_future_out = yb[..., 0].squeeze(-1)
            y_cls = (y_future_out > 0).float()
            logits = model.outage_logits_from_weather(xb).squeeze(-1)
            rate = model.outage_rate_from_weather(xb).squeeze(-1)
            spike = model.outage_spike_from_weather(xb).squeeze(-1)
            loss_cls = bce(logits, y_cls)
            pos_mask = y_cls.bool()
            if pos_mask.any():
                lam = rate[pos_mask] + F.relu(spike[pos_mask])
                loss_mag = F.poisson_nll_loss(lam + 1e-8, y_future_out[pos_mask], log_input=False)
                residual = F.relu(y_future_out[pos_mask] - rate[pos_mask])
                loss_spike = quantile_loss(spike[pos_mask], residual, QUANTILE_TAU)
            else:
                loss_mag = torch.tensor(0.0, device=xb.device)
                loss_spike = torch.tensor(0.0, device=xb.device)

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

            loss = (
                W_SSL * loss_ssl
                + W_FORECAST * loss_fore
                + W_CONTRAST * loss_contrast
                + (W_CLS * cls_scale) * loss_cls
                + W_MAG * loss_mag
                + W_SPIKE * loss_spike
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            epoch_loss += loss.item() * Bsz
            seen_samples += Bsz

        epoch_loss /= max(1, seen_samples)
        took = time.time() - start
        lr_str = ", ".join(f"{pg['lr']:.2e}" for pg in opt.param_groups)
        print(f"[Pretrain] Epoch {epoch:02d} total={epoch_loss:.6f} "
              f"ssl(tr)={loss_ssl.item():.6f} fore(tr)={loss_fore.item():.6f} "
              f"cls(tr)={loss_cls.item():.6f} mag(tr)={loss_mag.item():.6f} spike(tr)={loss_spike.item():.6f} "
              f"time={took:.2f}s | batches={len(loader)} samples={seen_samples} | LR={lr_str}")

        # ---------- Validation: per-loss + classifier ratios ----------
        (val_total, val_ssl, val_fore, val_cls, val_mag, val_spike,
         z2n, n2z, z2n0, n2z0) = compute_val_metrics()

        if np.isfinite(val_total):
            print(f"[Pretrain Val] Epoch {epoch:02d}: "
                  f"ValTotal={val_total:.6f} | SSL={val_ssl:.6f} | Forecast={val_fore:.6f} | "
                  f"Cls={val_cls:.6f} | Mag={val_mag:.6f} | Spike={val_spike:.6f}")
        else:
            print(f"[Pretrain Val] Epoch {epoch:02d}: ValTotal=nan | SSL=nan | Forecast=nan | Cls=nan | Mag=nan | Spike=nan")

        # ------- classification zero/nonzero ratios (validation) -------
        if np.isfinite(z2n) or np.isfinite(n2z):
            print(f"[Pretrain Val] Classifier ratios — Zero→Nonzero={z2n:.4f} (all-zero={z2n0:.4f}) | "
                  f"Nonzero→Zero={n2z:.4f} (all-zero={n2z0:.4f})")
        else:
            print(f"[Pretrain Val] Classifier ratios — Zero→Nonzero=nan (all-zero=nan) | "
                  f"Nonzero→Zero=nan (all-zero=nan)")

        scheduler.step()

        # ---------- Save best by BEST_BY ('loss' expected) ----------
        metric_map = {
            "loss": val_total
        }
        key = str(BEST_BY).lower()
        chosen = metric_map.get(key, val_total)

        def _is_better(new, old):
            if not np.isfinite(new): return False
            if not np.isfinite(old): return True
            return new < old

        ckpt_payload = {
            "cb_model_name": CB_MODEL_NAME,
            "state_dict": model.encoder.state_dict(),   # encoder weights (unchanged key)
            "d_out": D_OUT,
            "in_channels": Din,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "val_total": val_total,
            "val_ssl": val_ssl,
            "val_forecast": val_fore,
            "val_cls": val_cls,
            "val_mag": val_mag,
            "val_spike": val_spike,
            "z2n": z2n,
            "n2z": n2z,
            "best_by": key,
            "epoch": epoch,
            # save classifier head so finetuning can gate with it
            "cls_head_state_dict": model.cls_head.state_dict(),
            "rate_head_state_dict": model.rate_head.state_dict(),
            "spike_head_state_dict": model.spike_head.state_dict(),
        }

        if _is_better(chosen, best["loss"]):
            best["loss"] = chosen
            best["epoch"] = epoch
            torch.save(ckpt_payload, SAVE_PATH)
            print(f"[Pretrain Val] New best (loss={val_total:.6f}) at epoch {epoch}. Saved: {SAVE_PATH}")

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
    train_multitask_contrast(ds_train)
    ds_train.close()

if __name__ == "__main__":
    main()
