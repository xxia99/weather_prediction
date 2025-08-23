import os
import time
from math import ceil
from tqdm import tqdm
import numpy as np
import pandas as pd  # <-- added
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =========================
# Config
# =========================
SPLIT_DIR   = "data/"
OUT_DIR     = "results_seq/"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Windows / data
SEQ_LEN     = 24     # context length T (match downstream)
H_PRE       = 24     # pretraining forecast horizon H
BATCH_SIZE  = 16

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
USE_CONTRASTIVE         = True
W_CONTRAST              = 1.0      # weight of contrastive loss
TAU                     = 0.2      # temperature for InfoNCE
ADD_JITTER_STD          = 0.01     # small Gaussian jitter std for views
ADD_DROPOUT_PROB        = 0.0      # channel dropout prob for views
# If you want contrastive gradients to update the encoder, set to True
CONTRAST_UPDATES_ENCODER = False

# ---------- Frequency-domain augmentations (TimesNet-style) ----------
USE_FREQ_AUG            = True
FREQ_AUG_PROB           = 0.8      # chance to apply frequency aug per view
FREQ_AMP_STD            = 0.10     # multiplicative magnitude noise ~ N(1, std)
FREQ_PHASE_STD          = 0.20     # phase jitter (radians) ~ N(0, std)
FREQ_DROPOUT_FRAC       = 0.10     # fraction of frequency bins to zero
FREQ_DROPOUT_PROB       = 0.5      # prob of applying spectral dropout
FREQ_BAND_PROB          = 0.5      # prob of band-stop (else band-keep) op
FREQ_BAND_WIDTH_RATIO   = (0.05, 0.25)  # width as fraction of Nyquist [min, max]
FREQ_SHELF_PROB         = 0.3      # prob of applying high/low shelf attenuation
FREQ_SHELF_STRENGTH     = (0.3, 0.8)    # attenuation factor range

# ---------- Multitask loss weights ----------
W_SSL       = 1.0   # masked reconstruction loss weight
W_FORECAST  = 1.0   # next-H forecasting loss weight

# ---------- PatchTST encoder hyperparams ----------
PATCH_LEN      = 2
PATCH_STRIDE   = None   # None => stride=PATCH_LEN (non-overlap)
PATCH_D_MODEL  = 64
PATCH_NHEAD    = 4
PATCH_NLAYERS  = 2
PATCH_DROPOUT  = 0.3
PATCH_FF       = 4 * PATCH_D_MODEL

# ---------- Train/Val split by days (match CB script logic) ----------
TRAIN_LAST_DAY = 80      # train on days 1..80, validate on the rest

# ---------- Save best metric + path (match CB script behavior) ----------
# Choose which metric picks the best checkpoint: "loss" or "rmse"
SAVE_BEST_BY = "rmse"
SAVE_PATH = os.path.join(OUT_DIR, "pretrained_patchtst_validation.pt")

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
# Data prep: windows for SSL + forecasting
# =========================
def prepare_multitask_windows(ds_train, seq_len=24, horizon=24):
    """
    Returns:
      X     : (nW, T, N, Din)                      inputs (scaled)
      Y_fut : (nW, H, N, Din)                      future targets for all channels
      Din   : 1 (outage) + F (weather)
      scalers: dict
    """
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    T, N, F = w.shape
    Din = 1 + F

    # scale
    y_mu, y_sd = zfit(y.reshape(-1, 1))
    w_mu, w_sd = zfit(w.reshape(-1, F))
    y_sc = zapply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, N)        # (T, N)
    w_sc = zapply(w.reshape(-1, F), w_mu, w_sd).reshape(T, N, F)     # (T, N, F)

    # windows
    nW = T - seq_len - horizon + 1
    if nW <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)          # (T, N, Din)
    X     = np.stack([X_all[i:i+seq_len] for i in range(nW)], axis=0)          # (nW, T, N, Din)
    Y_fut = np.stack([X_all[i+seq_len:i+seq_len+horizon] for i in range(nW)], axis=0)  # (nW, H, N, Din)

    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    return X, Y_fut, Din, scalers


# =========================
# Split helpers (copied from CB script, unchanged)
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

def prepare_multitask_windows_split(ds_train, seq_len=24, horizon=24, train_last_day=80):
    """Build train windows using only targets fully within days <= train_last_day. Also return aux for rolling val."""
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, uniq_days, day_end_idx = _compute_day_indices(ts)

    T, N, F = w.shape
    Din = 1 + F

    # Fit scalers on training portion ONLY
    scalers = _fit_scalers_train_only(y, w, days, train_last_day)
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)  # (T, N, Din)

    # Total possible windows
    nW_total = T - seq_len - horizon + 1
    if nW_total <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    # Compute how many windows end at or before end of day train_last_day
    last_train_idx = day_end_idx[train_last_day - 1]
    nW_train = max(0, last_train_idx - seq_len - horizon + 1)

    X_list, Y_list = [], []
    for node in range(N):
        x_loc = X_all[:, node, :]  # (T, Din)
        Xw = np.stack([x_loc[i:i+seq_len] for i in range(nW_total)], axis=0)               # (nW_total, seq_len, Din)
        Yw = np.stack([X_all[i+seq_len:i+seq_len+horizon, node, :] for i in range(nW_total)], axis=0)  # (nW_total, H, Din)
        if nW_train > 0:
            X_list.append(Xw[:nW_train])
            Y_list.append(Yw[:nW_train])
        else:
            X_list.append(np.empty((0, seq_len, Din), dtype=float))
            Y_list.append(np.empty((0, horizon, Din), dtype=float))

    X_tr = np.stack(X_list, axis=2) if X_list else np.empty((0, seq_len, 0, Din))
    Y_tr = np.stack(Y_list, axis=2) if Y_list else np.empty((0, horizon, 0, Din))

    aux = {
        "X_all_scaled": X_all,     # (T, N, Din)
        "y_scaled": y_sc,          # (T, N)
        "timestamps": ts,          # (T,)
        "days": days,              # (T,)
        "day_end_idx": day_end_idx # list[int]
    }
    return X_tr, Y_tr, Din, scalers, aux

def build_val_windows_day_ahead(aux, horizon, seq_len, train_last_day=80):
    """Validation windows day d -> day d+1, identical to CB script."""
    X_all = aux["X_all_scaled"]           # (T, N, Din)
    ts    = aux["timestamps"]
    days  = aux["days"]
    day_end_idx = aux["day_end_idx"]
    T, N, Din = X_all.shape

    Xv, Yv = [], []
    last_day = int(days.max())
    for d in range(train_last_day, last_day):
        end_idx = day_end_idx[d - 1]           # last index of day d
        start_ctx = end_idx - (seq_len - 1)
        fut_start = end_idx + 1
        fut_end   = fut_start + horizon - 1
        if start_ctx < 0 or fut_end >= T:
            continue
        Xv.append(X_all[start_ctx:end_idx+1, :, :])          # (seq_len, N, Din)
        Yv.append(X_all[fut_start:fut_start+horizon, :, :])  # (horizon, N, Din)

    if not Xv:
        return (np.empty((0, seq_len, N, Din), dtype=float),
                np.empty((0, horizon, N, Din), dtype=float))
    Xv = np.stack(Xv, axis=0)  # (nD, seq_len, N, Din)
    Yv = np.stack(Yv, axis=0)  # (nD, horizon, N, Din)
    return Xv, Yv


# =========================
# PatchTST Encoder
# =========================
class PatchTSTEncoder(nn.Module):
    """
    Channel-last patching along time for each node independently.
    Input  : (B, T, N, F)
    Output : (B, T, N, D)  (token embedding repeated to steps and trimmed)
    """
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
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        P, S = self.patch_len, self.stride
        assert F == self.in_channels, f"PatchTST got F={F}, expected {self.in_channels}"

        # per-node
        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)

        # pad to fit patches
        L = (T + S - 1) // S
        T_pad = (L - 1) * S + P
        if T_pad < T:
            L += 1
            T_pad = (L - 1) * S + P
        if T_pad > T:
            pad = torch.zeros(B * N, T_pad - T, F, device=x.device, dtype=x.dtype)
            x_bn = torch.cat([x_bn, pad], dim=1)  # (B*N, T_pad, F)

        # unfold into patches
        x_unf = x_bn.unfold(dimension=1, size=P, step=S)          # (B*N, L, P, F)
        x_unf = x_unf.contiguous().view(B * N, L, P * F)          # (B*N, L, P*F)

        # tokens -> transformer
        tokens = self.proj(x_unf)                                 # (B*N, L, D)
        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)                              # (B*N, L, D)

        # unpatch by repeating each token across P steps, trim to T
        tokens_rep = tokens.repeat_interleave(P, dim=1)            # (B*N, L*P, D)
        steps = tokens_rep[:, :T, :]                               # (B*N, T, D)

        # back to (B, T, N, D)
        h = steps.view(B, N, T, self.d_model).permute(0, 2, 1, 3)
        return h


# =========================
# Multi-task pretrain model
# =========================
class MultiTaskPatchTST(nn.Module):
    """
    Encoder + heads:
      - recon_head: reconstruct inputs at masked timesteps (SSL)
      - forecast_head: predict next H steps of full multivariate inputs
      - proj_head: projection for InfoNCE (contrastive)
    """
    def __init__(self, in_channels, horizon):
        super().__init__()
        self.horizon = horizon
        self.encoder = PatchTSTEncoder(in_channels=in_channels)

        # SSL reconstruction head: (B, T, N, D) -> (B, T, N, F)
        self.recon_head = nn.Linear(PATCH_D_MODEL, in_channels)

        # Forecast head: mean-pool over time per node, then MLP -> (H * F)
        self.forecast_head = nn.Sequential(
            nn.Linear(PATCH_D_MODEL, 2 * PATCH_D_MODEL),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * PATCH_D_MODEL, horizon * in_channels)
        )

        # Projection head for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(PATCH_D_MODEL, PATCH_D_MODEL),
            nn.GELU(),
            nn.Linear(PATCH_D_MODEL, PATCH_D_MODEL)
        )

    def encode_steps(self, x):
        """Return per-step embeddings (B, T, N, D)."""
        return self.encoder(x)

    def forward(self, x):
        # x: (B, T, N, F)
        h = self.encoder(x)                    # (B, T, N, D)

        # Reconstruction at each past step
        recon = self.recon_head(h)             # (B, T, N, F)

        # Forecast next H steps for all channels
        pooled = h.mean(dim=1)                 # (B, N, D)
        fut = self.forecast_head(pooled)       # (B, N, H*F)
        B, N, _ = fut.shape
        fut = fut.view(B, N, self.horizon, -1).permute(0, 2, 1, 3)  # (B, H, N, F)

        return recon, fut

    def pooled_projection(self, x):
        """
        For contrastive: encode -> pool across T and N -> projection -> L2 normalize.
        If CONTRAST_UPDATES_ENCODER is False, we detach to avoid encoder grads here.
        """
        h = self.encoder(x) if CONTRAST_UPDATES_ENCODER else self.encoder(x).detach()
        pooled = h.mean(dim=1).mean(dim=1)       # (B, D)
        z = self.proj_head(pooled)               # (B, D)
        z = F.normalize(z, dim=-1)
        return z


# =========================
# Losses
# =========================
def masked_recon_loss(pred, target, mask):
    """
    pred/target: (B, T, N, F)
    mask:       (B, T, 1, 1) 1 where masked (reconstruct), 0 otherwise
    """
    m = mask.expand_as(pred)
    denom = m.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * m).sum() / denom

def forecast_loss(pred, target):
    """
    pred:   (B, H, N, F)
    target: (B, H, N, F)
    """
    return torch.mean((pred - target) ** 2)

def info_nce_loss(z1, z2, tau=0.2):
    """
    NT-Xent across batch. z1,z2: (B, D), L2-normalized.
    We compute both directions (z1->z2 and z2->z1) and average.
    """
    B = z1.size(0)
    if B == 1:
        return torch.tensor(0.0, device=z1.device)
    sim = (z1 @ z2.t()) / tau           # (B, B)
    labels = torch.arange(B, device=z1.device)
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_12 + loss_21)


# =========================
# Masking & Augmentations
# =========================
def make_temporal_mask(batch_size, T, mode="random", ratio=0.3, span_len=4, device="cpu"):
    """
    Returns mask of shape (B, T, 1, 1) with 1.0 at masked timesteps.
    - random: IID Bernoulli per step with prob=ratio.
    - span:   choose K spans of length span_len covering approx ratio of T.
    """
    if mode == "random":
        m = (torch.rand(batch_size, T, 1, 1, device=device) < ratio).float()
        return m

    # span mode
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
    """Zero-out masked timesteps. x: (B,T,N,F); mask: (B,T,1,1)."""
    return x * (1.0 - mask)


def add_jitter(x, std=0.01):
    """Add small Gaussian noise."""
    if std <= 0.0:
        return x
    noise = torch.randn_like(x) * std
    return x + noise


def dropout_features(x, p=0.0):
    """Randomly drop features per timestep (channel dropout)."""
    if p <= 0.0:
        return x
    B, T, N, F = x.shape
    mask = (torch.rand(B, 1, 1, F, device=x.device) < p).float()
    return x * (1.0 - mask)


# ====== Frequency-domain augmentations (TimesNet-style) ======
@torch.no_grad()
def _variance_match(x_aug, x_ref, eps=1e-6):
    """Rescale x_aug to match the per-sample std of x_ref."""
    # std over (T,N,F) for each batch element
    s_ref = x_ref.flatten(1).std(dim=1, keepdim=True).clamp_min(eps)          # (B,1)
    s_aug = x_aug.flatten(1).std(dim=1, keepdim=True).clamp_min(eps)          # (B,1)
    scale = (s_ref / s_aug).view(-1, 1, 1, 1)
    return x_aug * scale

def _random_band_mask(B, Lf, device, keep=False):
    """
    Create (B, Lf, 1, 1) band mask. If keep=False: band-stop; if keep=True: band-keep.
    """
    # choose center and width
    centers = torch.rand(B, device=device) * (Lf - 1)
    widths  = torch.empty(B, device=device).uniform_(
        int(FREQ_BAND_WIDTH_RATIO[0] * Lf),
        max(1.0, FREQ_BAND_WIDTH_RATIO[1] * Lf)
    )
    # frequency indices
    idx = torch.arange(Lf, device=device).view(1, Lf)  # (1, Lf)
    mask = torch.ones(B, Lf, 1, 1, device=device)
    for b in range(B):
        c = centers[b]
        w = widths[b]
        lo = (c - w / 2.0).clamp(0, Lf - 1)
        hi = (c + w / 2.0).clamp(0, Lf - 1)
        band = ((idx >= lo) & (idx <= hi)).float().view(1, Lf, 1, 1)
        if keep:
            mask[b:b+1] = band  # keep only band
        else:
            mask[b:b+1] = 1.0 - band  # zero band (stop)
    return mask

def _shelf_curve(Lf, high=True, strength=0.5, device="cpu"):
    """
    Build a high/low shelf attenuation curve over frequencies.
    Returns shape (1, Lf, 1, 1)
    """
    f = torch.linspace(0.0, 1.0, steps=Lf, device=device).view(1, Lf, 1, 1)
    if high:
        curve = 1.0 - strength * f.pow(1.0)  # attenuate high freq
    else:
        curve = 1.0 - strength * (1.0 - f).pow(1.0)  # attenuate low freq
    return curve.clamp_min(0.0)

def freq_augment(x, prob=FREQ_AUG_PROB):
    """
    Apply a mix of frequency-domain augs on x (B,T,N,F) with probability `prob`:
      - amplitude scaling in freq domain
      - phase jitter
      - band stop/keep
      - spectral dropout
      - shelving (high/low attenuation)
    Then normalize to match original variance.
    """
    if (not USE_FREQ_AUG) or (torch.rand(1).item() > prob):
        return x

    B, T, N, Fdim = x.shape
    # rFFT along time
    Xf = torch.fft.rfft(x, dim=1)  # (B, Lf, N, F), complex
    Lf = Xf.size(1)

    # amplitude scaling
    if FREQ_AMP_STD > 0.0:
        amp = torch.randn(B, Lf, 1, 1, device=x.device, dtype=Xf.real.dtype) * FREQ_AMP_STD + 1.0
        Xf = Xf * amp

    # phase jitter
    if FREQ_PHASE_STD > 0.0:
        phi = torch.randn(B, Lf, 1, 1, device=x.device, dtype=Xf.real.dtype) * FREQ_PHASE_STD
        rot = torch.complex(torch.cos(phi), torch.sin(phi))
        Xf = Xf * rot

    # band op (stop or keep)
    if torch.rand(1).item() < FREQ_BAND_PROB:
        keep = False
    else:
        keep = True
    band_mask = _random_band_mask(B, Lf, device=x.device, keep=keep)  # real mask
    Xf = Xf * band_mask

    # spectral dropout
    if torch.rand(1).item() < FREQ_DROPOUT_PROB and FREQ_DROPOUT_FRAC > 0.0:
        # choose bins to drop per-sample
        k = max(1, int(FREQ_DROPOUT_FRAC * Lf))
        drop = torch.zeros(B, Lf, 1, 1, device=x.device)
        for b in range(B):
            idx = torch.randperm(Lf, device=x.device)[:k]
            drop[b, idx] = 1.0
        Xf = Xf * (1.0 - drop)

    # high/low shelf
    if torch.rand(1).item() < FREQ_SHELF_PROB:
        strength = torch.empty(1, device=x.device).uniform_(*FREQ_SHELF_STRENGTH).item()
        high = bool(torch.randint(0, 2, (1,), device=x.device).item())
        shelf = _shelf_curve(Lf, high=high, strength=strength, device=x.device)  # (1,Lf,1,1)
        Xf = Xf * shelf

    # inverse FFT (ensure original length T)
    x_aug = torch.fft.irfft(Xf, n=T, dim=1)  # (B,T,N,F)
    # variance match (stabilize magnitude)
    x_aug = _variance_match(x_aug, x_ref=x)
    return x_aug


def make_view(x, mask_mode, mask_ratio, span_len, jitter_std, feat_drop):
    """
    Build one augmented view:
      - temporal mask (random/span),
      - optional feature dropout,
      - optional Gaussian jitter,
      - frequency-domain augmentations.
    """
    B, T, _, _ = x.shape
    m = make_temporal_mask(B, T, mode=mask_mode, ratio=mask_ratio, span_len=span_len, device=x.device)
    xv = apply_mask(x, m)
    xv = dropout_features(xv, p=feat_drop)
    xv = add_jitter(xv, std=jitter_std)
    xv = freq_augment(xv, prob=FREQ_AUG_PROB)
    return xv


# =========================
# Rolling-val helpers (copied from CB script, unchanged)
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

    y = ds_train.out.transpose("timestamp", "location").values.astype(float)  # (T, N)
    locs = list(map(str, ds_train.location.values))
    df = pd.DataFrame(y[fut_indices, :], columns=locs, index=ts_fut).reset_index().rename(columns={"index": "timestamp"})
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

@torch.no_grad()
def _predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day=80) -> pd.DataFrame:
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

        Xin = X_all[start_ctx:end_idx+1, :, :].reshape(1, SEQ_LEN, N, Din)
        xb = torch.tensor(Xin, dtype=torch.float32, device=DEVICE)

        recon, fut = model(xb)                  # fut: (1, H, N, F)
        fut_np = fut[0].detach().cpu().numpy()  # (H, N, F)

        # de-scale only the first channel (outage) for metric, same as CB code
        pred_sc = fut_np[:, :, 0].transpose(1, 0)  # (N, H)

        y_mu = float(scalers["y_mu"].flatten()[0])
        y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0
        pred = (pred_sc * y_sd) + y_mu
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

        ts_future = pd.to_datetime(ts[fut_start:fut_start + horizon])
        for i, county in enumerate(locs):
            rows.append(pd.DataFrame({
                "timestamp": ts_future,
                "location": county,
                "pred": pred[i].astype(float)
            }))

    if rows:
        return pd.concat(rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=["timestamp", "location", "pred"])


# =========================
# Training loop (now mirrors CB script’s val/saving/logging)
# =========================
def train_multitask_contrast(ds_train):
    # --- Build split-aware train windows (fit scalers on days 1..80) ---
    X_tr, Y_fut_tr, Din, scalers, aux = prepare_multitask_windows_split(
        ds_train, seq_len=SEQ_LEN, horizon=H_PRE, train_last_day=TRAIN_LAST_DAY
    )
    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                            torch.tensor(Y_fut_tr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Build validation windows for loss (day d -> day d+1) ---
    X_val, Y_val = build_val_windows_day_ahead(aux, horizon=H_PRE, seq_len=SEQ_LEN, train_last_day=TRAIN_LAST_DAY)
    have_val_loss = X_val.shape[0] > 0
    if have_val_loss:
        val_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Y_val, dtype=torch.float32))
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    else:
        val_loader = None

    # --- Model ---
    model = MultiTaskPatchTST(in_channels=Din, horizon=H_PRE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Prepare ground-truth for rolling validation RMSE once
    df_truth_val = _truth_wide_for_val(ds_train, aux, H_PRE, train_last_day=TRAIN_LAST_DAY)

    best = {"loss": float("inf"), "rmse": float("inf"), "epoch": -1}

    def compute_val_loss():
        if val_loader is None:
            return float("nan")
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)  # (B, T, N, F)
                yb = yb.to(DEVICE)  # (B, H, N, F)
                Bsz, T, _, _ = xb.shape
                # same masking as training for validation of composite loss
                mask_ssl = make_temporal_mask(
                    Bsz, T, mode=MASK_MODE, ratio=MASK_RATIO, span_len=SPAN_LEN, device=xb.device
                )
                xmasked = apply_mask(xb, mask_ssl)
                recon, fut = model(xmasked)
                l_ssl  = masked_recon_loss(recon, xb, mask_ssl)
                l_fore = forecast_loss(fut, yb)
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
                l = W_SSL * l_ssl + W_FORECAST * l_fore + W_CONTRAST * l_contrast
                total += l.item() * xb.size(0)
                count += xb.size(0)
        return total / max(1, count)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        seen_samples = 0  # track actual samples processed (mirrors CB log)
        start = time.time()

        for xb, yb in tqdm(loader, desc=f"Pretrain Epoch {epoch}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)      # (B, T, N, F)
            yb = yb.to(DEVICE)      # (B, H, N, F)

            Bsz, T, _, _ = xb.shape

            # ---------- SSL mask for reconstruction ----------
            mask_ssl = make_temporal_mask(
                Bsz, T, mode=MASK_MODE, ratio=MASK_RATIO, span_len=SPAN_LEN, device=xb.device
            )
            xmasked = apply_mask(xb, mask_ssl)

            # ---------- Forward (reconstruction + forecasting) ----------
            opt.zero_grad()
            recon, fut = model(xmasked)  # recon: (B,T,N,F), fut: (B,H,N,F)

            loss_ssl  = masked_recon_loss(recon, xb, mask_ssl)
            loss_fore = forecast_loss(fut, yb)

            # ---------- Contrastive branch ----------
            if USE_CONTRASTIVE:
                # Two independent augmented views
                x1 = make_view(
                    xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                    jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB
                )
                x2 = make_view(
                    xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                    jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB
                )
                z1 = model.pooled_projection(x1)  # (B, D)
                z2 = model.pooled_projection(x2)  # (B, D)
                loss_contrast = info_nce_loss(z1, z2, tau=TAU)
            else:
                loss_contrast = torch.tensor(0.0, device=xb.device)

            loss = W_SSL * loss_ssl + W_FORECAST * loss_fore + W_CONTRAST * loss_contrast

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            epoch_loss += loss.item() * Bsz
            seen_samples += Bsz

        # Average by actual samples (logging mirrors CB)
        epoch_loss /= max(1, seen_samples)
        took = time.time() - start
        print(f"[Pretrain] Epoch {epoch:02d} total={epoch_loss:.6f} "
              f"ssl={loss_ssl.item():.6f} fore={loss_fore.item():.6f} "
              f"contrast={loss_contrast.item():.6f} time={took:.2f}s | "
              f"batches={len(loader)} samples={seen_samples}")

        # ---------- Validation: loss ----------
        val_loss = compute_val_loss()

        # ---------- Validation: rolling RMSE ----------
        if df_truth_val is not None and not df_truth_val.empty:
            df_pred_val = _predict_val_roll(ds_train, model, scalers, H_PRE, aux, TRAIN_LAST_DAY)
            if not df_pred_val.empty:
                rmse_val = _per_county_rmse_mean(df_truth_val, df_pred_val)
            else:
                rmse_val = float("nan")
        else:
            rmse_val = float("nan")

        print(f"[Pretrain Val] Epoch {epoch:02d}: ValLoss={val_loss:.6f} | Rolling RMSE={rmse_val:.4f}")

        # ---------- Save best (same policy as CB script) ----------
        save_by_loss = (SAVE_BEST_BY.lower() == "loss")
        metric = val_loss if save_by_loss else rmse_val
        better = np.isfinite(metric) and (metric < best["loss"] if save_by_loss else metric < best["rmse"])
        if better:
            if save_by_loss:
                best["loss"] = metric
            else:
                best["rmse"] = metric
            best["epoch"] = epoch
            torch.save(
                {
                    "state_dict": model.encoder.state_dict(),
                    "in_channels": Din,
                    "patch_len": PATCH_LEN,
                    "patch_stride": PATCH_STRIDE,
                    "d_model": PATCH_D_MODEL,
                    "val_loss": val_loss,
                    "val_rmse": rmse_val,
                    "epoch": epoch,
                    "saved_by": SAVE_BEST_BY.lower(),
                },
                SAVE_PATH,
            )
            print(f"[Pretrain Val] New best ({SAVE_BEST_BY}={metric:.6f}) at epoch {epoch}. Saved: {SAVE_PATH}")

    # Note: best model already saved when improved.


# =========================
# Main
# =========================
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    train_multitask_contrast(ds_train)
    ds_train.close()

if __name__ == "__main__":
    main()
