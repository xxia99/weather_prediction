# demo_stgnn.py
# Spatial-Temporal Graph Neural Network forecaster for outage counts.
# Uses previous SEQ_LEN hours of outage + weather features across all counties
# to predict the next 24 hours of outages jointly for all counties.

import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- config ----------------
SPLIT_DIR = "data/"
OUT_DIR   = "results_stgnn/"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN    = 24
BATCH_SIZE = 8     # each sample is the full state of all counties
EPOCHS     = 5
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN     = 32
K_NEIGHBOR = 5     # neighbors per node when building correlation graph

# ---------------- utils ----------------
def open_ds(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path)

def zfit(arr):
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def zapply(arr, mu, sd):
    return (arr - mu) / sd

def build_corr_adjacency(y, k=5):
    """Build a normalized adjacency from pairwise outage correlations.

    Negative correlations and NaNs are clipped to zero so that the graph
    normalization step does not produce infinities or NaNs when computing the
    degree matrix.
    """
    L = y.shape[1]
    corr = np.corrcoef(y.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[corr < 0] = 0.0

    A = np.zeros((L, L), dtype=float)
    for i in range(L):
        idx = np.argsort(-corr[i])[:k]
        A[i, idx] = corr[i, idx]
    A = np.maximum(A, A.T)
    A = A + np.eye(L)
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-6))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)

def build_windows(X, y, seq_len, horizon):
    """Build sliding windows over the full graph."""
    T = y.shape[0]
    N = T - seq_len - horizon + 1
    Xw, Yw = [], []
    for i in range(N):
        Xw.append(X[i:i+seq_len])           # (seq_len, L, F)
        Yw.append(y[i+seq_len:i+seq_len+horizon].T)  # (L, horizon)
    return np.asarray(Xw, dtype=float), np.asarray(Yw, dtype=float)

# ---------------- model ----------------
class SimpleSTGNN(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, horizon, A):
        super().__init__()
        self.num_nodes = num_nodes
        self.A = A
        self.gconv = nn.Linear(in_dim, hidden_dim, bias=False)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        x = x.to(self.A.device)
        # Graph convolution on each timestep
        x = torch.einsum('ij,btjf->btif', self.A, x)  # (B, T, N, F)
        x = self.gconv(x)                             # (B, T, N, H)
        x = torch.relu(x)
        # collapse node dimension into batch for temporal modeling
        x = x.permute(0, 2, 1, 3).reshape(B*N, T, -1)
        _, h = self.rnn(x)                            # (1, B*N, H)
        h = h[-1]                                     # (B*N, H)
        out = self.head(h)                            # (B*N, horizon)
        out = out.reshape(B, N, -1)                   # (B, N, horizon)
        return out

# ---------------- pipeline helpers ----------------
def prepare_training(ds_train, horizon):
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)        # (T, L)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)  # (T, L, F)
    T, L, F = w.shape

    y_mu, y_sd = zfit(y.reshape(-1, 1))
    w_mu, w_sd = zfit(w.reshape(-1, F))
    y_sc = zapply(y.reshape(-1,1), y_mu, y_sd).reshape(T, L)
    w_sc = zapply(w.reshape(-1,F), w_mu, w_sd).reshape(T, L, F)

    Din = 1 + F
    X = np.concatenate([y_sc[..., None], w_sc], axis=2)  # (T, L, Din)
    Xw, Yw = build_windows(X, y_sc, SEQ_LEN, horizon)
    A = build_corr_adjacency(y_sc, k=K_NEIGHBOR)
    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    return Xw, Yw, Din, A, scalers

def train_model(X, Y, Din, A, horizon):
    if len(X) == 0:
        return None
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleSTGNN(num_nodes=A.size(0), in_dim=Din, hidden_dim=HIDDEN, horizon=horizon, A=A.to(DEVICE)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dl.dataset)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f}")
    return model

@torch.no_grad()
def predict_future(ds_train, model, scalers, A, horizon, ts_future):
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    T, L, F = w.shape
    y_sc = zapply(y.reshape(-1,1), scalers["y_mu"], scalers["y_sd"]).reshape(T, L)
    w_sc = zapply(w.reshape(-1,F), scalers["w_mu"], scalers["w_sd"]).reshape(T, L, F)
    X = np.concatenate([y_sc[...,None], w_sc], axis=2)
    Xin = X[-SEQ_LEN:].reshape(1, SEQ_LEN, L, -1)
    pred_sc = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE)).cpu().numpy()[0]  # (L, horizon)
    pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
    pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)
    locs = list(map(str, ds_train.location.values))
    rows = []
    for li, county in enumerate(locs):
        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred[li]}))
    return pd.concat(rows, ignore_index=True)

# ---------------- main ----------------
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    ds_test24 = open_ds(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    horizon = len(ts24)

    X, Y, Din, A, scalers = prepare_training(ds_train, horizon)
    model = train_model(X, Y, Din, A, horizon)

    if model is not None:
        df = predict_future(ds_train, model, scalers, A.to(DEVICE), horizon, ts24)
    else:
        locs = list(map(str, ds_train.location.values))
        df = pd.DataFrame({"timestamp": np.repeat(ts24, len(locs)),
                           "location": np.tile(locs, len(ts24)),
                           "pred": 0.0})

    out_path = os.path.join(OUT_DIR, "stgnn_pred_24h.csv")
    df.to_csv(out_path, index=False)
    print("Saved predictions:", out_path)

    ds_train.close(); ds_test24.close()

if __name__ == "__main__":
    main()
