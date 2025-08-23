# Simple shared Seq2Seq per-county forecaster (outage + ALL weather features).
# Transformer encoder over sliding windows from all counties, then predicts
# each county's next H hours using only the last SEQ_LEN hours of its own history.
# Mirrors the split-aware validation/testing/saving logic.

import os
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import xarray as xr
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from demo_evaluate import load_truth_wide, per_county_rmse_mean
from event_stratified_eval import event_stratified_report

# ---------------- config ----------------
SPLIT_DIR = "data/"
OUT_DIR   = "results_transformer/"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN      = 24
BATCH_SIZE   = 64
EPOCHS       = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL      = 128
NHEAD        = 8
NLAYERS      = 2
DROPOUT      = 0.1

# Day split: train on days 1..80, validate (rolling day-ahead) on the rest
TRAIN_LAST_DAY = 80

# ---------------- utils ----------------
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
    """
    Build (X,y) windows for one location.
    X_loc: (T, D), y_loc: (T,)
    Returns Xw: (N, seq_len, D), Yw: (N, horizon)
    """
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return np.empty((0, seq_len, X_loc.shape[1]), dtype=float), np.empty((0, horizon), dtype=float)
    Xw, Yw = [], []
    for i in range(N):
        Xw.append(X_loc[i:i+seq_len])
        Yw.append(y_loc[i+seq_len:i+seq_len+horizon])
    return np.asarray(Xw, dtype=float), np.asarray(Yw, dtype=float)

# ---------------- model ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (T, B, d_model)
        return x + self.pe[:x.size(0)]

class TransformerForecaster(nn.Module):
    def __init__(self, din, horizon):
        super().__init__()
        self.input_proj = nn.Linear(din, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=256, dropout=DROPOUT, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NLAYERS)
        self.pos_encoder = PositionalEncoding(D_MODEL)
        self.head = nn.Linear(D_MODEL, horizon)

    def forward(self, x):
        # x: (B, T, D)
        x = self.input_proj(x)            # (B, T, D_MODEL)
        x = x.transpose(0, 1)             # (T, B, D_MODEL)
        x = self.pos_encoder(x)
        enc = self.encoder(x)             # (T, B, D_MODEL)
        last = enc[-1]                    # (B, D_MODEL)
        return self.head(last)            # (B, horizon)

# ---------------- training / infer ----------------
def train_model(X, Y, din, horizon, eval_callback=None):
    if len(X) == 0:
        return None
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = TransformerForecaster(din=din, horizon=horizon).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s")
        if eval_callback is not None:
            eval_callback(model, epoch + 1)

    return model

@torch.no_grad()
def infer(model, Xin):
    model.eval()
    out = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE))
    return out.cpu().numpy()

# ---------------- split-aware helpers (train on days 1..80; val day-ahead on the rest) ----------------
def _compute_day_indices(ts: np.ndarray):
    """
    Map each timestamp to a day index starting at 1.
    Returns: days (T,), unique_days (list[pd.Timestamp]), day_end_idx (list[int])
    """
    ts_pd = pd.to_datetime(ts)
    days_norm = ts_pd.normalize()
    uniq = pd.DatetimeIndex(sorted(days_norm.unique()))
    day_map = {d: i+1 for i, d in enumerate(uniq)}
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

def prepare_training_split(ds_train, horizon, train_last_day=80):
    """
    Build training windows **only** whose targets lie fully within days <= train_last_day.
    Returns:
      X_tr: (num_windows_total, SEQ_LEN, Din)
      Y_tr: (num_windows_total, horizon)
      Din, scalers (fit on train split)
      aux: dict with arrays used for rolling validation (scaled series, timestamps, etc.)
    """
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, L)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, L, F)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, uniq_days, day_end_idx = _compute_day_indices(ts)
    T, L, F = w.shape
    Din = 1 + F

    # Scalers from training portion only
    scalers = _fit_scalers_train_only(y, w, days, train_last_day)

    # Scale all (using train-only scalers)
    y_sc = zapply(y.reshape(-1,1), scalers["y_mu"], scalers["y_sd"]).reshape(T, L)     # (T, L)
    w_sc = zapply(w.reshape(-1,F), scalers["w_mu"], scalers["w_sd"]).reshape(T, L, F)  # (T, L, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)                             # (T, L, Din)

    N_total = T - SEQ_LEN - horizon + 1
    if N_total <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    last_train_idx = day_end_idx[train_last_day - 1]
    N_train = max(0, last_train_idx - SEQ_LEN - horizon + 1)

    X_list, Y_list = [], []
    for li in range(L):
        y_loc = y_sc[:, li]         # (T,)
        X_loc = X_all[:, li, :]     # (T, Din)
        Xw, Yw = build_windows(X_loc, y_loc, SEQ_LEN, horizon)
        if len(Xw):
            X_list.append(Xw[:N_train])
            Y_list.append(Yw[:N_train])

    X_tr = np.concatenate(X_list, axis=0) if X_list else np.empty((0, SEQ_LEN, Din), dtype=float)
    Y_tr = np.concatenate(Y_list, axis=0) if Y_list else np.empty((0, horizon), dtype=float)

    aux = {
        "X_all_scaled": X_all,     # (T, L, Din)
        "y_scaled": y_sc,          # (T, L)
        "timestamps": ts,          # (T,)
        "days": days,              # (T,)
        "day_end_idx": day_end_idx # list[int]
    }
    return X_tr, Y_tr, Din, scalers, aux

# --------- rolling validation (day d -> day d+1) ----------
@torch.no_grad()
def predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day=80):
    """
    For each day d in [train_last_day, last_day-1], use last SEQ_LEN hours of day d
    per county to predict the next `horizon` hours starting at day d+1.
    Returns long df: [timestamp, location, pred]
    """
    model.eval()
    X_all = aux["X_all_scaled"]           # (T, L, Din)
    ts    = aux["timestamps"]
    days  = aux["days"]
    day_end_idx = aux["day_end_idx"]
    L = X_all.shape[1]
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

        for li in range(L):
            Xin = X_all[start_ctx:end_idx+1, li, :].reshape(1, SEQ_LEN, X_all.shape[2])  # (1, SEQ_LEN, Din)
            pred_sc = infer(model, Xin)[0]  # (H,)

            # invert scaling
            y_mu = float(scalers["y_mu"].flatten()[0])
            y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0
            pred = (pred_sc * y_sd) + y_mu
            pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

            rows.append(pd.DataFrame({
                "timestamp": ts_future,
                "location": locs[li],
                "pred": pred.astype(float)
            }))

    if rows:
        return pd.concat(rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=["timestamp", "location", "pred"])

def truth_wide_for_val(ds_train, aux, horizon, train_last_day=80):
    """
    Build truth wide dataframe for the union of all validation future timestamps.
    """
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

    y = ds_train.out.transpose("timestamp", "location").values.astype(float)  # (T, L)
    locs = list(map(str, ds_train.location.values))
    df = pd.DataFrame(y[fut_indices, :], columns=locs, index=ts_fut).reset_index().rename(columns={"index": "timestamp"})
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

# -------- helper for all-zero baseline on a given truth frame --------
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

# -------------------- COERCE TIMESTAMPS (helper) --------------------
def _coerce_ts64(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df['timestamp'] is datetime64[ns] (not object)."""
    if df is None or df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("datetime64[ns]")
    return df

# -------- unified callback (rolling val + test; save best by val RMSE) --------
def make_val_callback_rolling(ds_train, scalers, horizon, aux, label,
                              train_last_day=80,
                              ts_future_test=None, df_truth_test=None, zero_rmse_test=float("nan")):
    """
    Evaluate each epoch on rolling day-ahead validation (days 81..end) **and** on the test set.
    Prints:
      - Rolling RMSE=xxx.xxxx (all-zero RMSE=yyy.yyyy)
      - Testing RMSE=aaa.aaaa (all-zero RMSE=bbb.bbbb)
    Saves best checkpoint (by validation RMSE).
    """
    best = {"rmse": float("inf"), "epoch": -1}
    df_truth_val = truth_wide_for_val(ds_train, aux, horizon, train_last_day)

    # all-zero baseline for validation timestamps
    if not df_truth_val.empty:
        zero_rmse_val = _all_zero_rmse_for_truth(df_truth_val)
    else:
        zero_rmse_val = float("nan")

    def _callback(model, epoch):
        # ---- Validation (rolling) ----
        df_pred_val = predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day)

        # ---------- COERCE TIMESTAMPS BEFORE STRATIFIED REPORT ----------
        df_pred_val = _coerce_ts64(df_pred_val)
        df_truth_val_local = _coerce_ts64(df_truth_val)

        # Event-stratified RMSEs (Calm vs Stormy), after anchor
        try:
            event_stratified_report(
                ds_train,
                df_truth_val_local,
                df_pred_val,
                train_last_day=TRAIN_LAST_DAY,
                method="mean",
                storm_top_q=0.80,
                calm_bottom_q=0.50,
                calm_n=10,
                seed=42
            )
        except ValueError as e:
            # One more defensive attempt with explicit dtype alignment
            print(f"[Val {label}] Note: re-trying event_stratified_report after dtype coercion: {e}")
            df_pred_val["timestamp"] = pd.to_datetime(df_pred_val["timestamp"]).astype("datetime64[ns]")
            df_truth_val_local["timestamp"] = pd.to_datetime(df_truth_val_local["timestamp"]).astype("datetime64[ns]")
            event_stratified_report(
                ds_train,
                df_truth_val_local,
                df_pred_val,
                train_last_day=TRAIN_LAST_DAY,
                method="mean",
                storm_top_q=0.80,
                calm_bottom_q=0.50,
                calm_n=10,
                seed=42
            )

        if df_pred_val.empty or df_truth_val_local.empty:
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE=nan (all-zero RMSE=nan)")
        else:
            rmse = per_county_rmse_mean(df_truth_val_local, df_pred_val)
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE={rmse:.4f} (all-zero RMSE={zero_rmse_val:.4f})")

            # Save best by validation RMSE
            if np.isfinite(rmse) and rmse < best["rmse"]:
                best["rmse"] = rmse
                best["epoch"] = epoch
                save_path = os.path.join(OUT_DIR, f"best_seq2seq_{label}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "label": label,
                        "rmse": rmse,
                        "model_state_dict": model.state_dict(),
                    },
                    save_path,
                )
                print(f"[Val {label}] New best RMSE {rmse:.4f} at epoch {epoch}. Saved: {save_path}")

        # ---- Test set evaluation at each validation step ----
        if ts_future_test is not None and df_truth_test is not None and len(ts_future_test) > 0:
            df_pred_test = predict_per_county(ds_train, model, scalers, horizon, ts_future_test)
            rmse_test = per_county_rmse_mean(df_truth_test, df_pred_test)
            print(f"[Val {label}] Epoch {epoch}: Testing RMSE={rmse_test:.4f} (all-zero RMSE={zero_rmse_test:.4f})")

    return _callback

# ---------------- pipeline helpers (split-aware) ----------------
def prepare_training(ds_train, horizon):
    """
    Wrapper that builds split-aware training windows and aux info for rolling validation.
    """
    X, Y, Din, scalers, aux = prepare_training_split(ds_train, horizon, TRAIN_LAST_DAY)
    return X, Y, Din, scalers, aux

# ---------------- test-time prediction (unchanged) ----------------
def predict_per_county(ds_train, model, scalers, horizon, ts_future):
    """
    For each county, form the last SEQ_LEN inputs from train history only, then
    predict the next 'horizon' values.
    """
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, L)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, L, F)
    T, L, F = w.shape
    Din = 1 + F

    # Scale with train (split) scalers
    y_sc = zapply(y.reshape(-1,1), scalers["y_mu"], scalers["y_sd"]).reshape(T, L)
    w_sc = zapply(w.reshape(-1,F), scalers["w_mu"], scalers["w_sd"]).reshape(T, L, F)

    rows = []
    for li, county in enumerate(locs):
        if T < SEQ_LEN or model is None:
            pred_sc = np.zeros(horizon, dtype=float)
        else:
            y_loc = y_sc[:, li]
            w_loc = w_sc[:, li, :]
            X_loc = np.concatenate([y_loc.reshape(-1,1), w_loc], axis=1)  # (T, Din)
            Xin = X_loc[-SEQ_LEN:].reshape(1, SEQ_LEN, Din)
            pred_sc = infer(model, Xin)[0]  # (H,)

        # Invert scaling and clamp
        pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))

    return pd.concat(rows, ignore_index=True)

# ---------------- main ----------------
def main():
    # Load train + test
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))
    # ds_train = open_ds(os.path.join(SPLIT_DIR, "train_50days.nc"))
    # ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "valid_1day.nc"))
    # ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "valid_2days.nc"))

    # Ensure features exist (and implicitly use all of them)
    _ = get_all_features(ds_train)

    # Horizons and timestamps (for tests)
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    # Truth data for test evaluation
    # df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "valid_1day.nc"))
    # df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "valid_2days.nc"))
    df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))

    # Pre-compute all-zero RMSE for test sets
    zero_rmse_24 = _all_zero_rmse_for_truth(df_t24)
    zero_rmse_48 = _all_zero_rmse_for_truth(df_t48)

    # ==== 24h ====
    X24, Y24, Din, scalers24, aux24 = prepare_training(ds_train, horizon=H24)
    val_cb24 = make_val_callback_rolling(
        ds_train, scalers24, H24, aux24, label="24h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts24, df_truth_test=df_t24, zero_rmse_test=zero_rmse_24
    )
    mdl24 = train_model(X24, Y24, Din, H24, eval_callback=val_cb24)

    # ==== 48h ====
    X48, Y48, _, scalers48, aux48 = prepare_training(ds_train, horizon=H48)
    val_cb48 = make_val_callback_rolling(
        ds_train, scalers48, H48, aux48, label="48h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts48, df_truth_test=df_t48, zero_rmse_test=zero_rmse_48
    )
    mdl48 = train_model(X48, Y48, Din, H48, eval_callback=val_cb48)

    # ----- Test evaluation using best models, then write CSVs -----
    # 24h
    if mdl24 is not None:
        best24_path = os.path.join(OUT_DIR, "best_seq2seq_24h.pt")
        if os.path.exists(best24_path):
            ckpt24 = torch.load(best24_path, map_location=DEVICE)
            try:
                mdl24.load_state_dict(ckpt24["model_state_dict"])
                rmse24_val = ckpt24.get("rmse", None)
                ep24 = ckpt24.get("epoch", None)
                if isinstance(rmse24_val, (float, int)):
                    print(f"[24h] Loaded best checkpoint (epoch {ep24}, Val RMSE={rmse24_val:.4f})")
                else:
                    print(f"[24h] Loaded best checkpoint (epoch {ep24})")
            except Exception as e:
                print(f"[24h] Warning: failed to load best checkpoint: {e}")
        # Predict on test set and report with all-zero baseline
        df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24)
        test_rmse_24 = per_county_rmse_mean(df_t24, df24)
        print(f"[Test 24h] Testing RMSE={test_rmse_24:.4f} (all-zero RMSE={zero_rmse_24:.4f})")
    else:
        df24 = None

    # 48h
    if mdl48 is not None:
        best48_path = os.path.join(OUT_DIR, "best_seq2seq_48h.pt")
        if os.path.exists(best48_path):
            ckpt48 = torch.load(best48_path, map_location=DEVICE)
            try:
                mdl48.load_state_dict(ckpt48["model_state_dict"])
                rmse48_val = ckpt48.get("rmse", None)
                ep48 = ckpt48.get("epoch", None)
                if isinstance(rmse48_val, (float, int)):
                    print(f"[48h] Loaded best checkpoint (epoch {ep48}, Val RMSE={rmse48_val:.4f})")
                else:
                    print(f"[48h] Loaded best checkpoint (epoch {ep48})")
            except Exception as e:
                print(f"[48h] Warning: failed to load best checkpoint: {e}")
        # Predict on test set and report with all-zero baseline
        df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48)
        test_rmse_48 = per_county_rmse_mean(df_t48, df48)
        print(f"[Test 48h] Testing RMSE={test_rmse_48:.4f} (all-zero RMSE={zero_rmse_48:.4f})")
    else:
        df48 = None

    # Fallbacks (shouldn't trigger)
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

    # Save predictions
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
