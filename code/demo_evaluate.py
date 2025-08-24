# demo_evaluate_per_county.py
# Computes RMSE per county, then averages the RMSEs across counties.
# Also computes the RMSE for an all-zero prediction baseline.
# Expects per-county predictions (long format): ["timestamp","location","pred"].

import os
import numpy as np
import pandas as pd
import xarray as xr
from math import sqrt

SPLIT_DIR = "data/"

PRED_DIR  = "results/"
PRED_24 = os.path.join(PRED_DIR, "sarimax_pred_24h.csv")  # change if your per-county filename differs
PRED_48 = os.path.join(PRED_DIR, "sarimax_pred_48h.csv")  # change if your per-county filename differs

# PRED_DIR  = "results_seq/"
# PRED_24 = os.path.join(PRED_DIR, "seq2seq_pred_24h.csv")  # change if your per-county filename differs
# PRED_48 = os.path.join(PRED_DIR, "seq2seq_pred_48h.csv")  # change if your per-county filename differs
# PRED_24 = os.path.join(PRED_DIR, "stgnn_pred_24h.csv")  # change if your per-county filename differs
# PRED_48 = os.path.join(PRED_DIR, "stgnn_pred_48h.csv")  # change if your per-county filename differs
# PRED_24 = os.path.join(PRED_DIR, "stemgnn_pred_24h.csv")  # change if your per-county filename differs
# PRED_48 = os.path.join(PRED_DIR, "stemgnn_pred_48h.csv")
# PRED_DIR  = "results_stemgnn/"
# PRED_24 = os.path.join(PRED_DIR, "stemgnn_pred_24h.csv")  # change if your per-county filename differs
# PRED_48 = os.path.join(PRED_DIR, "seq2seq_pred_48h.csv")
#
# PRED_DIR  = "results_stgnn/"
# PRED_24 = os.path.join(PRED_DIR, "stgnn_pred_24h.csv")  # change if your per-county filename differs
# # PRED_48 = os.path.join(PRED_DIR, "seq2seq_pred_48h.csv")

def rmse(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))

def load_truth_wide(nc_path: str) -> pd.DataFrame:
    """Return wide df with columns: timestamp, <loc1>, <loc2>, ..."""
    if not os.path.exists(nc_path):
        raise FileNotFoundError(nc_path)
    ds = xr.open_dataset(nc_path)
    try:
        y = ds["out"].transpose("timestamp", "location").values.astype(float)  # (T, L)
        ts = pd.to_datetime(ds["timestamp"].values)
        # Make sure locations are strings to match CSVs reliably
        locs = list(map(str, ds["location"].values))
    finally:
        ds.close()
    df = pd.DataFrame(y, columns=locs, index=ts).reset_index().rename(columns={"index": "timestamp"})
    # De-duplicate timestamps just in case
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def load_pred_long(csv_path: str) -> pd.DataFrame:
    """Load per-county predictions in long format."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    req = {"timestamp", "location", "pred"}
    if not req.issubset(df.columns):
        raise ValueError(f"Prediction file must have columns {req}, got {list(df.columns)}")
    # Ensure location keys are strings to match NetCDF
    df["location"] = df["location"].astype(str)
    # Drop state-level rows if present
    df = df[df["location"] != "STATE_AGG"].copy()
    # De-duplicate
    df = df.drop_duplicates(subset=["timestamp", "location"]).sort_values(["timestamp", "location"]).reset_index(drop=True)
    # Coerce pred to float
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce").fillna(0.0)
    return df[["timestamp", "location", "pred"]]

def long_to_wide(df_long: pd.DataFrame, locs: list[str]) -> pd.DataFrame:
    """Pivot to wide; ensure all counties present as columns; fill missing preds with 0."""
    if df_long.empty:
        # produce an empty wide frame with requested columns
        return pd.DataFrame(columns=["timestamp", *locs])
    pivot = df_long.pivot_table(
        index="timestamp", columns="location", values="pred", aggfunc="sum", fill_value=0.0
    )
    # Ensure all target locations are present and in same order
    pivot = pivot.reindex(columns=locs, fill_value=0.0)
    pivot = pivot.sort_index().reset_index()
    # Columns become a MultiIndex name; normalize back to strings
    pivot.columns.name = None
    return pivot

def per_county_rmse_mean(df_truth_wide: pd.DataFrame, df_pred_long: pd.DataFrame) -> float:
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]

    # Copy and coerce timestamps to datetime for reliable merging
    truth = df_truth_wide.copy()
    pred_long = df_pred_long.copy()
    if "timestamp" in truth.columns:
        truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    if "timestamp" in pred_long.columns:
        pred_long["timestamp"] = pd.to_datetime(pred_long["timestamp"])

    df_pred_wide = long_to_wide(pred_long, locs)
    if "timestamp" in df_pred_wide.columns:
        df_pred_wide["timestamp"] = pd.to_datetime(df_pred_wide["timestamp"])

    # Evaluate on the intersection of timestamps that appear in both truth and preds
    df = truth.merge(df_pred_wide, on="timestamp", how="inner", suffixes=("_true", "_pred"))
    if df.empty:
        # No overlapping timestamps
        return float("nan")

    rmses = []
    for loc in locs:
        yt = df[f"{loc}_true"].values
        if f"{loc}_pred" in df.columns:
            yp = df[f"{loc}_pred"].values
        else:
            yp = np.zeros_like(yt)
        # Normalize
        if yt.max() <= 1:
            if yp.max() > yp.min():
                yp = (yp - yp.min()) / (yp.max() - yp.min())
            else:
                yp = np.zeros_like(yp)  # all values are the same

        rmses.append(rmse(yt, yp))
    return float(np.mean(rmses)) if rmses else float("nan")

def zero_baseline_rmse(df_truth_wide: pd.DataFrame) -> float:
    """Compute per-county RMSE mean for all-zero predictions."""
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]
    if not locs:
        return float("nan")
    rmses = []
    for loc in locs:
        y = df_truth_wide[loc].values
        p = np.ones_like(y, dtype=float)
        rmses.append(rmse(y, p))
    return float(np.mean(rmses))

def main():
    # Prefer demo tests; fall back to real tests if demos missing
    t24 = os.path.join(SPLIT_DIR, "test_24h_demo.nc")
    t48 = os.path.join(SPLIT_DIR, "test_48h_demo.nc")
    # t24 = os.path.join(SPLIT_DIR, "valid_1day.nc")
    # t48 = os.path.join(SPLIT_DIR, "valid_2days.nc")
    if not os.path.exists(t24):
        t24 = os.path.join(SPLIT_DIR, "test_24h.nc")
    if not os.path.exists(t48):
        t48 = os.path.join(SPLIT_DIR, "test_48h.nc")

    # Load truth (wide)
    df_t24 = load_truth_wide(t24)
    df_t48 = load_truth_wide(t48)

    # Load predictions (long, per-county)
    df_p24 = load_pred_long(PRED_24)
    df_p48 = load_pred_long(PRED_48)

    # Model RMSE (per-county first, then mean)
    r24 = per_county_rmse_mean(df_t24, df_p24)
    r48 = per_county_rmse_mean(df_t48, df_p48)
    r_avg = np.nanmean([r24, r48])

    # Zero baseline RMSE
    z24 = zero_baseline_rmse(df_t24)
    z48 = zero_baseline_rmse(df_t48)
    z_avg = np.nanmean([z24, z48])

    print("Per-county-first RMSE (mean across counties):")
    print(f"  24h: {r24:.4f} (zero baseline: {z24:.4f})")
    print(f"  48h: {r48:.4f} (zero baseline: {z48:.4f})")
    print(f"  Avg: {r_avg:.4f} (zero baseline: {z_avg:.4f})")

if __name__ == "__main__":
    main()
