# train_simple.py
# Trains a simple outages-only model per county (SARIMAX(1,0,1)),
# adds day-1..80 train / day-ahead rolling validation (24h & 48h),
# prints validation RMSEs (with all-zero baselines) plus calm/storm breakdowns,
# then writes per-county predictions for both horizons as CSVs and reports test RMSEs.

import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from demo_evaluate import load_truth_wide, per_county_rmse_mean

SPLIT_DIR = "data/"
OUT_DIR = "results_sarima/"
os.makedirs(OUT_DIR, exist_ok=True)

# Day split: train on days 1..80, validate (rolling day-ahead) on the rest
TRAIN_LAST_DAY = 80  # mirrors other scripts

# Event-stratified evaluation settings
EVENT_METHOD   = "mean"   # or "max"
STORM_TOP_Q    = 0.80     # top 20% days = stormy
CALM_BOTTOM_Q  = 0.50     # bottom 50% pool = calm
CALM_N         = 10       # sample 10 calm days
BEST_BY        = "overall"  # metric to select best


def _all_zero_rmse_for_truth(df_truth_wide: pd.DataFrame) -> float:
    """RMSE of an all-zero predictor against the given truth frame."""
    if df_truth_wide.empty:
        return float("nan")
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]
    df_zero_long = pd.DataFrame({
        "timestamp": np.repeat(df_truth_wide["timestamp"].values, len(locs)),
        "location": np.tile(locs, len(df_truth_wide)),
        "pred": 0.0,
    })
    return per_county_rmse_mean(df_truth_wide, df_zero_long)

def compute_day_indices(ts_np: np.ndarray):
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

def _per_day_outage_score(ds_train, method: str = "mean"):
    ts = ds_train.timestamp.values
    days, _, _ = compute_day_indices(ts)
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
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
    days, _, _ = compute_day_indices(ts_pd)
    mask = np.array([d in select_days for d in days], dtype=bool)
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

def _coerce_ts64(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df['timestamp'] is datetime64[ns]."""
    if df is None or df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("datetime64[ns]")
    return df


def event_rmse_blocks(ds_train, df_truth_val, df_pred_val,
                      train_last_day, method, storm_top_q, calm_bottom_q, calm_n, seed):
    rmse_overall = per_county_rmse_mean(df_truth_val, df_pred_val) if (not df_truth_val.empty and not df_pred_val.empty) else float("nan")
    zero_overall = _all_zero_rmse_for_truth(df_truth_val) if not df_truth_val.empty else float("nan")

    rng = np.random.default_rng(seed)
    ts = ds_train.timestamp.values
    days, day_score = _per_day_outage_score(ds_train, method=method)
    last_day = int(days.max())
    tgt_days = np.arange(train_last_day + 1, last_day + 1, dtype=int)
    if len(tgt_days) == 0:
        return rmse_overall, zero_overall, float("nan"), float("nan"), 0, float("nan"), float("nan"), 0

    tgt_scores = day_score[tgt_days - 1]
    storm_thr = np.quantile(tgt_scores, storm_top_q)
    calm_thr  = np.quantile(tgt_scores, calm_bottom_q)

    storm_days = set(tgt_days[tgt_scores >= storm_thr].tolist())
    calm_pool  = tgt_days[tgt_scores <= calm_thr].tolist()
    if len(calm_pool) > 0:
        k = int(min(calm_n, len(calm_pool)))
        calm_days = set(rng.choice(calm_pool, size=k, replace=False).tolist())
    else:
        calm_days = set()

    ts_storm = _timestamps_for_days(ts, storm_days)
    ts_calm  = _timestamps_for_days(ts, calm_days)

    if len(ts_calm) > 0:
        truth_calm, pred_calm = _filter_truth_pred_by_days(df_truth_val, df_pred_val, ts_calm)
        rmse_calm = per_county_rmse_mean(truth_calm, pred_calm)
        zero_calm = _all_zero_rmse_for_truth(truth_calm)
        n_calm = len(calm_days)
    else:
        rmse_calm, zero_calm, n_calm = float("nan"), float("nan"), 0

    if len(ts_storm) > 0:
        truth_storm, pred_storm = _filter_truth_pred_by_days(df_truth_val, df_pred_val, ts_storm)
        rmse_storm = per_county_rmse_mean(truth_storm, pred_storm)
        zero_storm = _all_zero_rmse_for_truth(truth_storm)
        n_storm = len(storm_days)
    else:
        rmse_storm, zero_storm, n_storm = float("nan"), float("nan"), 0

    return rmse_overall, zero_overall, rmse_calm, zero_calm, n_calm, rmse_storm, zero_storm, n_storm

def maybe_save_best(model, epoch, label, rmse_overall, rmse_calm, rmse_storm,
                    best, out_dir, prefix, best_by="overall"):
    metric_map = {"overall": rmse_overall, "calm": rmse_calm, "storm": rmse_storm}
    sel = metric_map.get(str(best_by).lower(), rmse_overall)
    sel_val = sel if np.isfinite(sel) else rmse_overall

    if np.isfinite(sel_val) and sel_val < best.get("rmse", float("inf")):
        best["rmse"] = sel_val
        best["epoch"] = epoch
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{prefix}_{label}.pt")
        state = model.state_dict() if (model is not None and hasattr(model, "state_dict")) else None
        torch.save({
            "epoch": epoch,
            "label": label,
            "rmse_select": sel_val,
            "rmse_overall": rmse_overall,
            "rmse_calm": rmse_calm,
            "rmse_storm": rmse_storm,
            "best_by": str(best_by).lower(),
            "model_state_dict": state,
        }, save_path)
        print(f"[Val {label}] New best ({best_by}) RMSE {sel_val:.4f} at epoch {epoch}. Saved: {save_path}")

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

def truth_wide_for_val(ds_train, horizon, train_last_day=80) -> pd.DataFrame:
    """
    Build the truth wide dataframe restricted to the union of all validation future timestamps.
    """
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, _, day_end_idx = compute_day_indices(ts)

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
    days, _, day_end_idx = compute_day_indices(ts)
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
    df_truth_val_24 = _coerce_ts64(truth_wide_for_val(ds_train, horizon=H24, train_last_day=TRAIN_LAST_DAY))
    df_pred_val_24  = _coerce_ts64(predict_val_roll_sarimax(ds_train, horizon=H24, train_last_day=TRAIN_LAST_DAY))
    if not df_truth_val_24.empty and not df_pred_val_24.empty:
        (rmse_overall_24, zero_val_24,
         rmse_calm_24, zero_calm_24, n_calm_24,
         rmse_storm_24, zero_storm_24, n_storm_24) = event_rmse_blocks(
            ds_train, df_truth_val_24, df_pred_val_24,
            train_last_day=TRAIN_LAST_DAY,
            method=EVENT_METHOD,
            storm_top_q=STORM_TOP_Q,
            calm_bottom_q=CALM_BOTTOM_Q,
            calm_n=CALM_N,
            seed=42,
        )

        print(f"[Val 24h] Rolling RMSE={rmse_overall_24:.4f} (all-zero RMSE={zero_val_24:.4f})")
        if np.isfinite(rmse_calm_24):
            print(f"[EventEval 24h] Calm-heavy ({n_calm_24} days): RMSE={rmse_calm_24:.4f} (all-zero={zero_calm_24:.4f})")
        else:
            print("[EventEval 24h] Calm-heavy: RMSE=nan (all-zero=nan)")
        if np.isfinite(rmse_storm_24):
            print(f"[EventEval 24h] Storm-heavy ({n_storm_24} days): RMSE={rmse_storm_24:.4f} (all-zero={zero_storm_24:.4f})")
        else:
            print("[EventEval 24h] Storm-heavy: RMSE=nan (all-zero=nan)")

        best24 = {"rmse": float("inf"), "epoch": -1}
        maybe_save_best(None, 0, "24h", rmse_overall_24, rmse_calm_24, rmse_storm_24,
                        best24, OUT_DIR, "best_sarima", BEST_BY)
    else:
        print("[Val 24h] Rolling RMSE=nan (all-zero RMSE=nan)")
        print("[EventEval 24h] Calm-heavy: RMSE=nan (all-zero=nan)")
        print("[EventEval 24h] Storm-heavy: RMSE=nan (all-zero=nan)")

    # 48h validation
    df_truth_val_48 = _coerce_ts64(truth_wide_for_val(ds_train, horizon=H48, train_last_day=TRAIN_LAST_DAY))
    df_pred_val_48  = _coerce_ts64(predict_val_roll_sarimax(ds_train, horizon=H48, train_last_day=TRAIN_LAST_DAY))
    if not df_truth_val_48.empty and not df_pred_val_48.empty:
        (rmse_overall_48, zero_val_48,
         rmse_calm_48, zero_calm_48, n_calm_48,
         rmse_storm_48, zero_storm_48, n_storm_48) = event_rmse_blocks(
            ds_train, df_truth_val_48, df_pred_val_48,
            train_last_day=TRAIN_LAST_DAY,
            method=EVENT_METHOD,
            storm_top_q=STORM_TOP_Q,
            calm_bottom_q=CALM_BOTTOM_Q,
            calm_n=CALM_N,
            seed=42,
        )

        print(f"[Val 48h] Rolling RMSE={rmse_overall_48:.4f} (all-zero RMSE={zero_val_48:.4f})")
        if np.isfinite(rmse_calm_48):
            print(f"[EventEval 48h] Calm-heavy ({n_calm_48} days): RMSE={rmse_calm_48:.4f} (all-zero={zero_calm_48:.4f})")
        else:
            print("[EventEval 48h] Calm-heavy: RMSE=nan (all-zero=nan)")
        if np.isfinite(rmse_storm_48):
            print(f"[EventEval 48h] Storm-heavy ({n_storm_48} days): RMSE={rmse_storm_48:.4f} (all-zero={zero_storm_48:.4f})")
        else:
            print("[EventEval 48h] Storm-heavy: RMSE=nan (all-zero=nan)")

        best48 = {"rmse": float("inf"), "epoch": -1}
        maybe_save_best(None, 0, "48h", rmse_overall_48, rmse_calm_48, rmse_storm_48,
                        best48, OUT_DIR, "best_sarima", BEST_BY)
    else:
        print("[Val 48h] Rolling RMSE=nan (all-zero RMSE=nan)")
        print("[EventEval 48h] Calm-heavy: RMSE=nan (all-zero=nan)")
        print("[EventEval 48h] Storm-heavy: RMSE=nan (all-zero=nan)")

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
