# train_simple.py
# Trains a simple outages-only model per county (SARIMAX(1,0,1)),
# adds day-1..80 train / day-ahead rolling validation (24h & 48h),
# prints validation RMSEs (with all-zero baselines), then writes
# per-county predictions for both horizons as CSVs and reports test RMSEs.

import os
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from demo_evaluate import load_truth_wide, per_county_rmse_mean

SPLIT_DIR = "data/"
OUT_DIR = "results/"
os.makedirs(OUT_DIR, exist_ok=True)

# Day split: train on days 1..80, validate (rolling day-ahead) on the rest
TRAIN_LAST_DAY = 88  # mirrors other scripts

def safe_fit_sarimax(y: np.ndarray):
    y = np.asarray(y, dtype=float).flatten()
    if len(y) < 8 or np.allclose(y, y[0]):
        return None
    try:
        model = SARIMAX(y, order=(1,0,1), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        return res
    except Exception:
        return None

# -------- helpers to mirror day-split validation --------
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

def truth_wide_for_val(ds_train, horizon, train_last_day=80) -> pd.DataFrame:
    """
    Build the truth wide dataframe restricted to the union of all validation future timestamps.
    """
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, _, day_end_idx = _compute_day_indices(ts)

    fut_indices = []
    last_day = int(days.max())
    for d in range(train_last_day, last_day):
        end_idx = day_end_idx[d - 1]
        fut_start = end_idx + 1
        fut_end = fut_start + horizon - 1
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

def predict_val_roll_sarimax(ds_train, horizon, train_last_day=80) -> pd.DataFrame:
    """
    For each county, fit SARIMAX on data up to end of day 'train_last_day'.
    Produce a single long forecast for the remaining timeline and select only
    the timestamps used by day-ahead validation (union over all days).
    Returns long df: [timestamp, location, pred]
    """
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, _, day_end_idx = _compute_day_indices(ts)
    last_train_end = day_end_idx[train_last_day - 1]
    fut_start = last_train_end + 1
    if fut_start >= len(ts):
        return pd.DataFrame(columns=["timestamp", "location", "pred"])

    # Build set of validation timestamps (union of day-ahead windows)
    fut_indices = []
    last_day = int(days.max())
    for d in range(train_last_day, last_day):
        end_idx = day_end_idx[d - 1]
        fs = end_idx + 1
        fe = fs + horizon - 1
        if fe < len(ts):
            fut_indices.extend(list(range(fs, fs + horizon)))
    if not fut_indices:
        return pd.DataFrame(columns=["timestamp", "location", "pred"])
    fut_indices = sorted(set(fut_indices))
    ts_need = pd.to_datetime(ts[fut_indices])

    # Forecast per county using only train portion (1..TRAIN_LAST_DAY)
    rows = []
    for county in ds_train.location.values:
        y_full = ds_train.out.sel(location=county).values.astype(float).flatten()
        y_train = y_full[:last_train_end + 1]
        res = safe_fit_sarimax(y_train)
        if res is None:
            # zeros for all required timestamps
            pred_full = np.zeros(max(0, len(ts) - fut_start), dtype=float)
        else:
            steps_total = max(0, len(ts) - fut_start)
            pred_full = np.asarray(res.forecast(steps=steps_total), dtype=float)

        # pick only needed timestamps
        idx_offsets = np.array(fut_indices) - fut_start
        valid_mask = (idx_offsets >= 0) & (idx_offsets < len(pred_full))
        sel = idx_offsets[valid_mask]
        preds = np.zeros(len(ts_need), dtype=float)
        preds[valid_mask] = pred_full[sel]

        rows.append(pd.DataFrame({
            "timestamp": ts_need,
            "location": str(county),
            "pred": preds
        }))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp", "location", "pred"])

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

def main():
    ds_train = xr.open_dataset(os.path.join(SPLIT_DIR, "train.nc"))
    # ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h.nc"))
    # ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h.nc"))
    ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))
    # ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "valid_1day.nc"))
    # ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "valid_2days.nc"))

    counties = list(ds_train.location.values)
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    # -------- Rolling validation (mirrors other models) --------
    # 24h validation
    df_truth_val_24 = truth_wide_for_val(ds_train, horizon=H24, train_last_day=TRAIN_LAST_DAY)
    df_pred_val_24  = predict_val_roll_sarimax(ds_train, horizon=H24, train_last_day=TRAIN_LAST_DAY)
    if not df_truth_val_24.empty and not df_pred_val_24.empty:
        zero_val_24 = _all_zero_rmse_for_truth(df_truth_val_24)
        rmse_val_24 = per_county_rmse_mean(df_truth_val_24, df_pred_val_24)
        print(f"[Val 24h] Rolling RMSE={rmse_val_24:.4f} (all-zero RMSE={zero_val_24:.4f})")
    else:
        print("[Val 24h] Rolling RMSE=nan (all-zero RMSE=nan)")

    # 48h validation
    df_truth_val_48 = truth_wide_for_val(ds_train, horizon=H48, train_last_day=TRAIN_LAST_DAY)
    df_pred_val_48  = predict_val_roll_sarimax(ds_train, horizon=H48, train_last_day=TRAIN_LAST_DAY)
    if not df_truth_val_48.empty and not df_pred_val_48.empty:
        zero_val_48 = _all_zero_rmse_for_truth(df_truth_val_48)
        rmse_val_48 = per_county_rmse_mean(df_truth_val_48, df_pred_val_48)
        print(f"[Val 48h] Rolling RMSE={rmse_val_48:.4f} (all-zero RMSE={zero_val_48:.4f})")
    else:
        print("[Val 48h] Rolling RMSE=nan (all-zero RMSE=nan)")

    # -------- Per-county test forecasts (same outputs as before) --------
    rows24 = []
    rows48 = []

    for county in counties:
        y_train = ds_train.out.sel(location=county).values.astype(float).flatten()
        res = safe_fit_sarimax(y_train)

        if res is None:
            pred24 = np.zeros(len(ts24), dtype=float)
            pred48 = np.zeros(len(ts48), dtype=float)
        else:
            pred48 = np.asarray(res.forecast(steps=len(ts48)), dtype=float)
            pred24 = pred48[:len(ts24)]

        rows24.append(pd.DataFrame({
            "timestamp": ts24,
            "location": county,
            "pred": pred24
        }))
        rows48.append(pd.DataFrame({
            "timestamp": ts48,
            "location": county,
            "pred": pred48
        }))

    # Concatenate & save
    df24 = pd.concat(rows24, ignore_index=True)
    df48 = pd.concat(rows48, ignore_index=True)

    out24 = os.path.join(OUT_DIR, "sarimax_pred_24h.csv")
    out48 = os.path.join(OUT_DIR, "sarimax_pred_48h.csv")
    df24.to_csv(out24, index=False)
    df48.to_csv(out48, index=False)

    print("Saved predictions:")
    print(f"  {out24}")
    print(f"  {out48}")

    # -------- Test RMSEs (mirror other scripts; use demo_evaluate) --------
    df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))
    zero_rmse_24 = _all_zero_rmse_for_truth(df_t24)
    zero_rmse_48 = _all_zero_rmse_for_truth(df_t48)

    test_rmse_24 = per_county_rmse_mean(df_t24, df24)
    test_rmse_48 = per_county_rmse_mean(df_t48, df48)
    print(f"[Test 24h] Testing RMSE={test_rmse_24:.4f} (all-zero RMSE={zero_rmse_24:.4f})")
    print(f"[Test 48h] Testing RMSE={test_rmse_48:.4f} (all-zero RMSE={zero_rmse_48:.4f})")

    # Clean up
    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
