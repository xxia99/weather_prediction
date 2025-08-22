import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- config ----------------
SPLIT_DIR  = "data/"
OUT_DIR    = "results_seq/"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Windows / data
SEQ_LEN    = 24          # same as your forecasting context
BATCH_SIZE = 32

# Training
EPOCHS     = 100
LR         = 1e-4
WEIGHT_DECAY = 1e-3

# Masking (self-supervised)
MASK_RATIO = 0.50        # fraction of timesteps to mask per window
TEMPORAL_MASK_SHARED = True  # mask a timestep across all nodes & features

# PatchTST encoder hyperparams (must match your forecasting script)
PATCH_LEN      = 2
PATCH_STRIDE   = None      # None => stride=PATCH_LEN (non-overlap)
PATCH_D_MODEL  = 64
PATCH_NHEAD    = 4
PATCH_NLAYERS  = 2
PATCH_DROPOUT  = 0.3
PATCH_FF       = 4 * PATCH_D_MODEL

SAVE_PATH = os.path.join(OUT_DIR, "pretrained_patchtst.pt")


# ---------------- utils ----------------
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

def build_windows(X_loc, y_loc, seq_len, horizon):
    """Build aligned windows for one location; horizon is unused in SSL, but we keep signature consistent."""
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return np.empty((0, seq_len, X_loc.shape[1]), dtype=float)
    Xw = []
    for i in range(N):
        Xw.append(X_loc[i:i+seq_len])
    return np.asarray(Xw, dtype=float)

def prepare_ssl_windows(ds_train, seq_len=24):
    """
    Returns:
      X: (num_windows, SEQ_LEN, N, Din)  with Din = 1 (outage) + F (weather)
      scalers: dict with y_mu, y_sd, w_mu, w_sd (not strictly needed for SSL, but kept for reference)
    """
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    T, N, F = w.shape

    # scale features
    y_mu, y_sd = zfit(y.reshape(-1, 1))
    w_mu, w_sd = zfit(w.reshape(-1, F))
    y_sc = zapply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, N)     # (T, N)
    w_sc = zapply(w.reshape(-1, F), w_mu, w_sd).reshape(T, N, F)  # (T, N, F)

    Din = 1 + F
    # build windows per node, then stack across nodes -> (nW, SEQ_LEN, N, Din)
    X_list = []
    horizon_dummy = 1  # not used, just to reuse the window builder
    for node in range(N):
        x_loc_full = np.concatenate([y_sc[:, node:node+1], w_sc[:, node, :]], axis=1)  # (T, Din)
        Xw = build_windows(x_loc_full, y_sc[:, node], seq_len, horizon_dummy)          # (nW, SEQ_LEN, Din)
        if len(Xw):
            X_list.append(Xw)

    if not X_list:
        raise RuntimeError("No SSL windows could be built (check SEQ_LEN vs data length).")

    # Stack across nodes: (nW, SEQ_LEN, N, Din)
    X = np.stack(X_list, axis=2)  # node axis
    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    return X, Din, scalers


# ---------------- PatchTST encoder ----------------
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


# ---------------- SSL model: encoder + prediction head ----------------
class MaskedStepPredictor(nn.Module):
    """
    Self-supervised model: PatchTST encoder + linear head to reconstruct inputs.
    Loss is computed only on masked timesteps.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = PatchTSTEncoder(in_channels=in_channels)
        self.head = nn.Linear(PATCH_D_MODEL, in_channels)

    def forward(self, x):
        # x: (B, T, N, F)
        h = self.encoder(x)                   # (B, T, N, D)
        y = self.head(h)                      # (B, T, N, F)
        return y


# ---------------- training loop ----------------
def masked_loss(pred, target, mask):
    """
    pred/target: (B, T, N, F)  on device
    mask:       (B, T, 1, 1)   1.0 where masked (to be predicted), else 0
    """
    # broadcast mask across nodes & features
    m = mask.expand_as(pred)
    num = (m.sum()).item()
    if num < 1.0:  # rare case if no timestep was sampled
        return torch.mean((pred - target) ** 2)
    return ((pred - target) ** 2 * m).sum() / (m.sum())


def train_ssl(ds_train):
    # Build windows
    X, Din, _ = prepare_ssl_windows(ds_train, seq_len=SEQ_LEN)  # (nW, T, N, Din)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MaskedStepPredictor(in_channels=Din).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for (xb,) in tqdm(dl, desc=f"SSL Epoch {epoch}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)  # (B, T, N, F)
            # Create temporal mask per-sample: (B, T, 1, 1)
            if TEMPORAL_MASK_SHARED:
                # mask specific timesteps across all nodes & features
                with torch.no_grad():
                    # choose ~MASK_RATIO of timesteps to drop per sample
                    B, T, N, F = xb.shape
                    # Bernoulli per time step
                    mask = (torch.rand(B, T, 1, 1, device=xb.device) < MASK_RATIO).float()
                    xmasked = xb * (1.0 - mask)  # zero-out masked timesteps
            else:
                # (optional alternative) independent mask per (t, n, f)
                with torch.no_grad():
                    mask = (torch.rand_like(xb[:, :, :, :1]) < MASK_RATIO).float()  # (B, T, N, 1)
                    xmasked = xb * (1.0 - mask)

                # unify mask shape to (B, T, 1, 1) for the loss
                mask = mask.mean(dim=2, keepdim=True)  # crude merge
                mask = (mask > 0).float()

            opt.zero_grad()
            yhat = model(xmasked)  # (B, T, N, F)
            loss = masked_loss(yhat, xb, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        print(f"[SSL] Epoch {epoch:02d}  Loss={epoch_loss:.6f}  Time={time.time()-start:.2f}s")

    # Save ONLY the encoder weights so your forecasting script can load them
    torch.save({"state_dict": model.encoder.state_dict()}, SAVE_PATH)
    print(f"[SSL] Saved pretrained encoder to: {SAVE_PATH}")


# ---------------- main ----------------
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    train_ssl(ds_train)
    ds_train.close()

if __name__ == "__main__":
    main()
