import os
import numpy as np
import pandas as pd
import torch
from demo_evaluate import per_county_rmse_mean


def compute_day_indices(ts_np: np.ndarray):
    """Map timestamps to day indices starting at 1.

    Returns
    -------
    days : np.ndarray
        Day index for each timestamp.
    uniq_days : list[pd.Timestamp]
        Sorted unique day boundaries.
    day_end_idx : list[int]
        Index of the last timestamp for each day.
    """
    ts_pd = pd.to_datetime(ts_np)
    days_norm = ts_pd.normalize()
    uniq = pd.DatetimeIndex(sorted(days_norm.unique()))
    day_map = {d: i + 1 for i, d in enumerate(uniq)}
    days = np.array([day_map[d] for d in days_norm], dtype=int)
    day_end_idx = []
    for d in uniq:
        idxs = np.where(days_norm == d)[0]
        day_end_idx.append(int(idxs.max()))
    return days, list(uniq), day_end_idx


def all_zero_rmse_for_truth(df_truth_wide: pd.DataFrame) -> float:
    """RMSE of an all-zero predictor for the given truth wide frame."""
    if df_truth_wide.empty:
        return float("nan")
    locs = [c for c in df_truth_wide.columns if c != "timestamp"]
    df_zero_long = pd.DataFrame({
        "timestamp": np.repeat(df_truth_wide["timestamp"].values, len(locs)),
        "location": np.tile(locs, len(df_truth_wide)),
        "pred": 0.0,
    })
    return per_county_rmse_mean(df_truth_wide, df_zero_long)


def _per_day_outage_score(ds_train, method: str = "mean"):
    """Return day indices and outage score per day."""
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
    days, _uniq, _end = compute_day_indices(ts_pd)
    mask = np.array([d in select_days for d in days], dtype=bool)
    return pd.DatetimeIndex(ts_pd[mask])


def _filter_truth_pred_by_days(df_truth_wide: pd.DataFrame,
                               df_pred_long: pd.DataFrame,
                               ts_subset: pd.DatetimeIndex):
    df_truth_sel = df_truth_wide[df_truth_wide["timestamp"].isin(ts_subset)].copy()
    df_pred_sel = df_pred_long[df_pred_long["timestamp"].isin(ts_subset)].copy()
    if not df_truth_sel.empty:
        df_truth_sel["timestamp"] = pd.to_datetime(df_truth_sel["timestamp"])
    if not df_pred_sel.empty:
        df_pred_sel["timestamp"] = pd.to_datetime(df_pred_sel["timestamp"])
    return df_truth_sel, df_pred_sel


def event_rmse_blocks(ds_train, df_truth_val, df_pred_val,
                      train_last_day, method, storm_top_q,
                      calm_bottom_q, calm_n, seed):
    """Compute overall, calm-day, and storm-day RMSE blocks."""
    truth_val = df_truth_val.copy()
    pred_val = df_pred_val.copy()
    if "timestamp" in truth_val.columns:
        truth_val["timestamp"] = pd.to_datetime(truth_val["timestamp"], errors="coerce")
    if "timestamp" in pred_val.columns:
        pred_val["timestamp"] = pd.to_datetime(pred_val["timestamp"], errors="coerce")

    rmse_overall = per_county_rmse_mean(truth_val, pred_val) if (not truth_val.empty and not pred_val.empty) else float("nan")
    zero_overall = all_zero_rmse_for_truth(truth_val) if not truth_val.empty else float("nan")

    rng = np.random.default_rng(seed)
    ts = ds_train.timestamp.values
    days, day_score = _per_day_outage_score(ds_train, method=method)
    last_day = int(days.max())
    tgt_days = np.arange(train_last_day + 1, last_day + 1, dtype=int)
    if len(tgt_days) == 0:
        return rmse_overall, zero_overall, float("nan"), float("nan"), 0, float("nan"), float("nan"), 0

    tgt_scores = day_score[tgt_days - 1]
    storm_thr = np.quantile(tgt_scores, storm_top_q)
    calm_thr = np.quantile(tgt_scores, calm_bottom_q)

    storm_days = set(tgt_days[tgt_scores >= storm_thr].tolist())
    calm_pool = tgt_days[tgt_scores <= calm_thr].tolist()
    if len(calm_pool) > 0:
        k = int(min(calm_n, len(calm_pool)))
        calm_days = set(rng.choice(calm_pool, size=k, replace=False).tolist())
    else:
        calm_days = set()

    ts_storm = _timestamps_for_days(ts, storm_days)
    ts_calm = _timestamps_for_days(ts, calm_days)

    if len(ts_calm) > 0:
        truth_calm, pred_calm = _filter_truth_pred_by_days(truth_val, pred_val, ts_calm)
        if not truth_calm.empty and not pred_calm.empty:
            rmse_calm = per_county_rmse_mean(truth_calm, pred_calm)
            zero_calm = all_zero_rmse_for_truth(truth_calm)
            n_calm = len(calm_days)
        else:
            rmse_calm, zero_calm, n_calm = float("nan"), float("nan"), 0
    else:
        rmse_calm, zero_calm, n_calm = float("nan"), float("nan"), 0

    if len(ts_storm) > 0:
        truth_storm, pred_storm = _filter_truth_pred_by_days(truth_val, pred_val, ts_storm)
        if not truth_storm.empty and not pred_storm.empty:
            rmse_storm = per_county_rmse_mean(truth_storm, pred_storm)
            zero_storm = all_zero_rmse_for_truth(truth_storm)
            n_storm = len(storm_days)
        else:
            rmse_storm, zero_storm, n_storm = float("nan"), float("nan"), 0
    else:
        rmse_storm, zero_storm, n_storm = float("nan"), float("nan"), 0

    return rmse_overall, zero_overall, rmse_calm, zero_calm, n_calm, rmse_storm, zero_storm, n_storm


def maybe_save_best(model, epoch, label, rmse_overall, rmse_calm, rmse_storm,
                    best, out_dir, prefix, best_by="overall"):
    """Update and save the best model checkpoint according to selected metric."""
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
