# demo_seq2seq.py
# Simple shared Seq2Seq per-county forecaster (outage + ALL weather features).
# Trains one LSTM encoder over sliding windows from all counties, then predicts
# each county's next H hours using only the last SEQ_LEN hours of its own history.
# Outputs per-county CSVs for 24h and 48h horizons.

import os
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from demo_evaluate import per_county_rmse_mean

# ---------------- config ----------------
SPLIT_DIR = "data/"
OUT_DIR   = "results_seq/"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN    = 24
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DHIDDEN    = 64
NLAYERS    = 1
VAL_DAYS   = 5  # hold out last 5 days for validation

# ---------------- utils ----------------
def open_ds(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path)

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

# ---------------- model ----------------
class SimpleSeq2Seq(nn.Module):
    def __init__(self, din, dh=64, nl=1, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(din, dh, nl, batch_first=True)
        self.head = nn.Linear(dh, horizon)
    def forward(self, x):
        # x: (B, T, D)
        _, (h, _) = self.lstm(x)
        hlast = h[-1]               # (B, dh)
        return self.head(hlast)     # (B, H)

def train_model(Xtr, Ytr, din, horizon, scalers, val_data=None):
    """Train the Seq2Seq model and report validation RMSE after each epoch."""
    if len(Xtr) == 0:
        return None

    ds_tr = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                          torch.tensor(Ytr, dtype=torch.float32))
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleSeq2Seq(din=din, dh=DHIDDEN, nl=NLAYERS, horizon=horizon).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl_tr, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl_tr.dataset)

        val_rmse = float("nan")
        if val_data is not None:
            model.eval()
            Xin = torch.tensor(val_data["Xin"], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                pred_sc = model(Xin).cpu().numpy()  # (L, H)
            pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
            pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

            ts = val_data["ts"]
            locs = val_data["locs"]
            rows = []
            for li, county in enumerate(locs):
                rows.append(pd.DataFrame({"timestamp": ts, "location": county, "pred": pred[li].astype(float)}))
            df_pred = pd.concat(rows, ignore_index=True)
            val_rmse = per_county_rmse_mean(val_data["truth"], df_pred)

        elapsed = time.time() - start_time
        if np.isnan(val_rmse):
            print(f"[Epoch {epoch+1}/{EPOCHS}] TrainLoss: {epoch_loss:.6f} | Time: {elapsed:.2f}s")
        else:
            print(f"[Epoch {epoch+1}/{EPOCHS}] TrainLoss: {epoch_loss:.6f} | ValRMSE: {val_rmse:.6f} | Time: {elapsed:.2f}s")

    return model

@torch.no_grad()
def infer(model, Xin):
    model.eval()
    out = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE))
    return out.cpu().numpy()

# ---------------- pipeline helpers ----------------
def prepare_train_val(ds, horizon, val_days=VAL_DAYS):
    """Split dataset into training windows and validation helpers.

    The last ``val_days`` of the series are reserved for validation. Scaling is
    fit on the training portion only. Validation metrics will forecast the first
    ``horizon`` hours of the held-out range using the final ``SEQ_LEN`` hours of
    history preceding it.
    """
    y = ds.out.transpose("timestamp", "location").values.astype(float)                 # (T, L)
    w = ds.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, L, F)
    ts = pd.to_datetime(ds.timestamp.values)
    T, L, F = w.shape

    val_hours = val_days * 24
    val_start = max(T - val_hours, 0)

    # Fit scalers on training portion only
    y_mu, y_sd = zfit(y[:val_start].reshape(-1, 1))
    w_mu, w_sd = zfit(w[:val_start].reshape(-1, F))

    # Scale entire arrays
    y_sc = zapply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, L)
    w_sc = zapply(w.reshape(-1, F), w_mu, w_sd).reshape(T, L, F)

    Din = 1 + F
    N = val_start - SEQ_LEN - horizon + 1
    X_tr, Y_tr = [], []
    if N > 0:
        for li in range(L):
            y_loc = y_sc[:, li]
            w_loc = w_sc[:, li, :]
            X_loc = np.concatenate([y_loc.reshape(-1, 1), w_loc], axis=1)
            for i in range(N):
                X_tr.append(X_loc[i:i + SEQ_LEN])
                Y_tr.append(y_loc[i + SEQ_LEN:i + SEQ_LEN + horizon])

    Xtr = np.asarray(X_tr, dtype=float) if X_tr else np.empty((0, SEQ_LEN, Din), dtype=float)
    Ytr = np.asarray(Y_tr, dtype=float) if Y_tr else np.empty((0, horizon), dtype=float)

    # Precompute validation helpers
    ts_val = ts[val_start:val_start + horizon]
    locs = list(map(str, ds.location.values))
    y_val = y[val_start:val_start + horizon]
    df_truth = pd.DataFrame(y_val, columns=locs)
    df_truth.insert(0, "timestamp", ts_val)

    Xin = []
    for li in range(L):
        X_loc = np.concatenate([y_sc[:, li].reshape(-1, 1), w_sc[:, li, :]], axis=1)
        hist = X_loc[max(0, val_start - SEQ_LEN):val_start]
        if hist.shape[0] < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - hist.shape[0], Din))
            hist = np.concatenate([pad, hist], axis=0)
        Xin.append(hist)
    Xin = np.asarray(Xin, dtype=float)  # (L, SEQ_LEN, Din)

    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    val_data = {"Xin": Xin, "ts": ts_val, "truth": df_truth, "locs": locs}
    return Xtr, Ytr, Din, scalers, val_data

def predict_per_county(ds_train, model, scalers, horizon, ts_future):
    """
    For each county, form the last SEQ_LEN inputs from train history only, then
    predict the next 'horizon' values. If insufficient history or model is None,
    output zeros for that county.
    """
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, L)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, L, F)
    T, L, F = w.shape
    Din = 1 + F

    # Scale with train scalers
    y_sc = zapply(y.reshape(-1,1), scalers["y_mu"], scalers["y_sd"]).reshape(T, L)
    w_sc = zapply(w.reshape(-1,F), scalers["w_mu"], scalers["w_sd"]).reshape(T, L, F)

    rows = []
    for li, county in enumerate(locs):
        # Build last SEQ_LEN inputs for this county
        if T < SEQ_LEN or model is None:
            pred_sc = np.zeros(horizon, dtype=float)
        else:
            y_loc = y_sc[:, li]
            w_loc = w_sc[:, li, :]
            X_loc = np.concatenate([y_loc.reshape(-1,1), w_loc], axis=1)  # (T, Din)
            Xin = X_loc[-SEQ_LEN:].reshape(1, SEQ_LEN, Din)
            pred_sc = infer(model, Xin)[0]  # (H,)

        # Invert scaling for y
        pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]

        # >>> non-negative clamp (and NaN guard)
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))

    return pd.concat(rows, ignore_index=True)

# ---------------- main ----------------
def main():
    # Load train + test
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    # ds_test24 = open_ds(os.path.join(SPLIT_DIR, "test_24h.nc"))
    # ds_test48 = open_ds(os.path.join(SPLIT_DIR, "test_48h.nc"))
    ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))

    # Ensure features exist (and implicitly use all of them)
    _ = get_all_features(ds_train)

    # Horizons and timestamps from tests (length only, no weather leakage)
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    # Prepare train/val windows and train models
    X24_tr, Y24_tr, Din, scalers24, val24 = prepare_train_val(ds_train, horizon=H24)
    mdl24 = train_model(X24_tr, Y24_tr, Din, H24, scalers24, val_data=val24)

    X48_tr, Y48_tr, Din, scalers48, val48 = prepare_train_val(ds_train, horizon=H48)
    mdl48 = train_model(X48_tr, Y48_tr, Din, H48, scalers48, val_data=val48)

    # Predict per county
    df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24) if mdl24 is not None else None
    df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48) if mdl48 is not None else None

    # Fallback to zeros if no windows (extremely short train)
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

    # Save 
    out24 = os.path.join(OUT_DIR, "seq2seq_pred_24h.csv")
    out48 = os.path.join(OUT_DIR, "seq2seq_pred_48h.csv")
    df24.to_csv(out24, index=False)
    df48.to_csv(out48, index=False)

    print("Saved predictions:")
    print(f"  {out24}")
    print(f"  {out48}")

    # Clean up
    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
