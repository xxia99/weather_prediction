import os
import time
from math import ceil
from tqdm import tqdm
import numpy as np
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

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Windows / data
SEQ_LEN     = 24     # context length T (match downstream)
H_PRE       = 24     # pretraining forecast horizon H
BATCH_SIZE  = 32

# Training
EPOCHS          = 20
LR              = 1e-4
WEIGHT_DECAY    = 1e-3
GRAD_CLIP_NORM  = 5.0

# ---------- Masking for SSL ----------
# Choose masking mode for SSL reconstruction:
MASK_MODE               = "span"   # "random" or "span"
MASK_RATIO              = 0.30     # fraction of timesteps masked per window
SPAN_LEN                = 4        # used if MASK_MODE=="span": contiguous length per span

# For contrastive views we can optionally use different (possibly lighter) masking:
VIEW_MASK_MODE          = "span"   # "random" or "span"
VIEW_MASK_RATIO         = 0.15
VIEW_SPAN_LEN           = 3

# ---------- Contrastive (InfoNCE) ----------
USE_CONTRASTIVE         = True
W_CONTRAST              = 1.0      # weight of contrastive loss
TAU                     = 0.2      # temperature for InfoNCE
ADD_JITTER_STD          = 0.01     # small Gaussian jitter std for contrastive views
ADD_DROPOUT_PROB        = 0.0      # set >0 to randomly drop features per timestep in views

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

# Save path
SAVE_PATH = os.path.join(OUT_DIR, "pretrained_patchtst_multitask_contrast.pt")


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
      scalers: dict (not strictly required, but useful to keep)
    """
    # y: outages (T, N); w: weather (T, N, F)
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    T, N, F = w.shape
    Din = 1 + F

    # scale
    y_mu, y_sd = zfit(y.reshape(-1, 1))
    w_mu, w_sd = zfit(w.reshape(-1, F))
    y_sc = zapply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, N)        # (T, N)
    w_sc = zapply(w.reshape(-1, F), w_mu, w_sd).reshape(T, N, F)     # (T, N, F)

    # global number of windows (same for all nodes)
    nW = T - seq_len - horizon + 1
    if nW <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    # X_all: (T, N, Din)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)

    # vectorized windows
    X = np.stack([X_all[i:i+seq_len] for i in range(nW)], axis=0)         # (nW, T, N, Din)
    Y_fut = np.stack([X_all[i+seq_len:i+seq_len+horizon]
                      for i in range(nW)], axis=0)                         # (nW, H, N, Din)

    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    return X, Y_fut, Din, scalers


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
        Returns z in (B, D).
        """
        with torch.no_grad():
            h = self.encoder(x)                  # (B, T, N, D)
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
        # avoid degenerate case
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


def make_view(x, mask_mode, mask_ratio, span_len, jitter_std, feat_drop):
    """
    Build one augmented view:
      - temporal mask (random/span),
      - optional feature dropout (same across T,N),
      - optional Gaussian jitter.
    """
    B, T, _, _ = x.shape
    m = make_temporal_mask(B, T, mode=mask_mode, ratio=mask_ratio, span_len=span_len, device=x.device)
    xv = apply_mask(x, m)
    xv = dropout_features(xv, p=feat_drop)
    xv = add_jitter(xv, std=jitter_std)
    return xv


# =========================
# Training loop
# =========================
def train_multitask_contrast(ds_train):
    # Build windows: X (nW, T, N, F), Y_fut (nW, H, N, F)
    X, Y_fut, Din, _ = prepare_multitask_windows(ds_train, seq_len=SEQ_LEN, horizon=H_PRE)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(Y_fut, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultiTaskPatchTST(in_channels=Din, horizon=H_PRE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
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
                # Encode -> pooled projections (no grad inside encode_steps via helper)
                # (We use pooled_projection which wraps encoder in no_grad to decouple it from main grads;
                #  If you want contrastive gradients to flow through encoder as well, remove no_grad in pooled_projection.)
                z1 = model.pooled_projection(x1)  # (B, D)
                z2 = model.pooled_projection(x2)  # (B, D)
                loss_contrast = info_nce_loss(z1, z2, tau=TAU)
            else:
                loss_contrast = torch.tensor(0.0, device=xb.device)

            loss = W_SSL * loss_ssl + W_FORECAST * loss_fore + W_CONTRAST * loss_contrast

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(loader.dataset)
        took = time.time() - start
        print(f"[Pretrain] Epoch {epoch:02d} total={epoch_loss:.6f} "
              f"ssl={loss_ssl.item():.6f} fore={loss_fore.item():.6f} "
              f"contrast={loss_contrast.item():.6f} time={took:.2f}s")

    # Save ONLY the encoder weights for downstream init
    torch.save({"state_dict": model.encoder.state_dict()}, SAVE_PATH)
    print(f"[Pretrain] Saved encoder to: {SAVE_PATH}")


# =========================
# Main
# =========================
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    train_multitask_contrast(ds_train)
    ds_train.close()

if __name__ == "__main__":
    main()
