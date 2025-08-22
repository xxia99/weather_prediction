import os
import time
from tqdm import tqdm
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
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

# Masked-step SSL
MASK_RATIO              = 0.30   # fraction of timesteps masked per window
TEMPORAL_MASK_SHARED    = True   # mask shared across nodes & features at a timestep

# Loss weights
W_SSL       = 1.0   # masked reconstruction loss weight
W_FORECAST  = 1.0   # next-H forecasting loss weight

# PatchTST encoder hyperparams (must align with downstream script)
PATCH_LEN      = 2
PATCH_STRIDE   = None   # None => stride=PATCH_LEN (non-overlap)
PATCH_D_MODEL  = 64
PATCH_NHEAD    = 4
PATCH_NLAYERS  = 2
PATCH_DROPOUT  = 0.3
PATCH_FF       = 4 * PATCH_D_MODEL

SAVE_PATH = os.path.join(OUT_DIR, "pretrained_patchtst_multitask.pt")


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

    # Build X and Y_fut jointly for all nodes without Python lists (vectorized)
    # X_all: (T, N, Din)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)

    # Collect windows by strided indexing
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
    Encoder + two heads:
      - recon_head: reconstruct inputs at masked timesteps (SSL)
      - forecast_head: predict next H steps of full multivariate inputs
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

    def forward(self, x):
        # x: (B, T, N, F)
        h = self.encoder(x)                    # (B, T, N, D)

        # Reconstruction at each past step
        recon = self.recon_head(h)             # (B, T, N, F)

        # Forecast next H steps for all channels
        pooled = h.mean(dim=1)                 # (B, N, D)
        fut = self.forecast_head(pooled)       # (B, N, H*F)
        B, N, _ = fut.shape
        # reshape to (B, H, N, F)
        fut = fut.view(B, N, self.horizon, -1).permute(0, 2, 1, 3)
        return recon, fut


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


# =========================
# Training loop
# =========================
def train_multitask(ds_train):
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

            # Build temporal mask: (B, T, 1, 1)
            if TEMPORAL_MASK_SHARED:
                with torch.no_grad():
                    Bsz, T, N, F = xb.shape
                    mask = (torch.rand(Bsz, T, 1, 1, device=xb.device) < MASK_RATIO).float()
                    xmasked = xb * (1.0 - mask)
            else:
                with torch.no_grad():
                    randmask = (torch.rand_like(xb[..., :1]) < MASK_RATIO).float()  # (B, T, N, 1)
                    xmasked = xb * (1.0 - randmask)
                    mask = (randmask.mean(dim=2, keepdim=True) > 0).float()

            opt.zero_grad()
            recon, fut = model(xmasked)                  # recon: (B, T, N, F), fut: (B, H, N, F)

            loss_ssl  = masked_recon_loss(recon, xb, mask)
            loss_fore = forecast_loss(fut, yb)
            loss = W_SSL * loss_ssl + W_FORECAST * loss_fore

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(loader.dataset)
        print(f"[Pretrain] Epoch {epoch:02d}  total={epoch_loss:.6f}  "
              f"ssl={loss_ssl.item():.6f}  fore={loss_fore.item():.6f}  "
              f"time={time.time()-start:.2f}s")

    # Save ONLY the encoder weights for downstream init
    torch.save({"state_dict": model.encoder.state_dict()}, SAVE_PATH)
    print(f"[Pretrain] Saved encoder to: {SAVE_PATH}")


# =========================
# Main
# =========================
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    train_multitask(ds_train)
    ds_train.close()

if __name__ == "__main__":
    main()
