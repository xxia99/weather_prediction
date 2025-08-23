# event_stratified_eval.py
import numpy as np
import pandas as pd
from typing import Tuple, Set
from demo_evaluate import per_county_rmse_mean


def _coerce_ts64(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df['timestamp'] is datetime64[ns] (not object/str)."""
    if df is None or df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("datetime64[ns]")
    return df


def _compute_day_indices(ts: np.ndarray) -> Tuple[np.ndarray, pd.DatetimeIndex, list[int]]:
    """Map each timestamp to a day index starting at 1, and return last index per day."""
    ts_pd = pd.to_datetime(ts)
    days_norm = ts_pd.normalize()
    uniq = pd.DatetimeIndex(sorted(days_norm.unique()))
    day_map = {d: i + 1 for i, d in enumerate(uniq)}
    days = np.array([day_map[d] for d in days_norm], dtype=int)
    day_end_idx = []
    for d in uniq:
        idxs = np.where(days_norm == d)[0]
        day_end_idx.append(int(idxs.max()))
    return days, uniq, day_end_idx


def _per_day_outage_score(ds_train, method: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      day_ids:  (T,) day index per timestamp (starts at 1)
      day_score: (max_day,) per-day score (mean or max across locations)
    """
    ts = ds_train.timestamp.values
    days, uniq_days, day_end_idx = _compute_day_indices(ts)
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)  # (T, N)

    # Aggregate outage per day across counties
    df = pd.DataFrame({"day": days, "y": y.mean(axis=1)})  # mean across locations at each timestamp
    # Then per-day aggregate
    if method.lower() == "max":
        g = df.groupby("day")["y"].max()
    else:
        g = df.groupby("day")["y"].mean()

    # Make dense array day -> score (1..max_day)
    max_day = int(days.max())
    day_score = np.zeros(max_day, dtype=float)
    day_score[g.index.values - 1] = g.values
    return days, day_score


def _timestamps_for_days(ts: np.ndarray, select_days: Set[int]) -> pd.DatetimeIndex:
    """
    Return the timestamps whose day-id (1-indexed) is in select_days.
    (Fixes previous bug where dtype/logic could drift.)
    """
    ts_pd = pd.to_datetime(ts)
    day_ids, _, _ = _compute_day_indices(ts_pd)
    if not select_days:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    mask = np.isin(day_ids, list(select_days))
    return pd.DatetimeIndex(ts_pd[mask])


def _filter_truth_pred_by_days(df_truth_wide: pd.DataFrame,
                               df_pred_long: pd.DataFrame,
                               ts_subset: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter both frames to the selected timestamps, with robust
    timestamp dtype alignment to avoid merge errors downstream.
    """
    df_truth_wide = _coerce_ts64(df_truth_wide)
    df_pred_long  = _coerce_ts64(df_pred_long)

    # Ensure subset index is datetime64[ns]
    ts_subset = pd.to_datetime(ts_subset).astype("datetime64[ns]")

    # Truth wide: filter rows by timestamp membership
    df_truth_sel = df_truth_wide[df_truth_wide["timestamp"].isin(ts_subset)].copy()
    # Pred long: filter rows by timestamp membership
    df_pred_sel = df_pred_long[df_pred_long["timestamp"].isin(ts_subset)].copy()

    # Coerce again just in case any copies reverted dtype
    df_truth_sel = _coerce_ts64(df_truth_sel)
    df_pred_sel  = _coerce_ts64(df_pred_sel)
    return df_truth_sel, df_pred_sel


def event_stratified_report(ds_train,
                            df_truth_val: pd.DataFrame,
                            df_pred_val: pd.DataFrame,
                            train_last_day: int,
                            method: str = "mean",
                            storm_top_q: float = 0.80,
                            calm_bottom_q: float = 0.50,
                            calm_n: int = 10,
                            seed: int = 42):
    """
    Prints RMSE on:
      - Calm-heavy: random 10 calm days AFTER the train anchor (days > train_last_day)
      - Storm-heavy: ALL stormy days AFTER the anchor
    Days are ranked by per-day outage score (mean or max across locations).

    This version fixes dtype mismatch issues by coercing timestamps to datetime64[ns].
    """
    if df_truth_val is None or df_pred_val is None or df_truth_val.empty or df_pred_val.empty:
        print("[EventEval] No validation predictions/truth to score.")
        return

    # Coerce once up-front to avoid downstream merge errors in per_county_rmse_mean.
    df_truth_val = _coerce_ts64(df_truth_val)
    df_pred_val  = _coerce_ts64(df_pred_val)

    rng = np.random.default_rng(seed)
    ts = ds_train.timestamp.values
    days, day_score = _per_day_outage_score(ds_train, method=method)

    # Target (future) days are train_last_day+1 .. last_day
    last_day = int(days.max())
    if train_last_day >= last_day:
        print("[EventEval] Anchor is at/after last day; nothing to evaluate.")
        return

    tgt_days = np.arange(train_last_day + 1, last_day + 1, dtype=int)
    tgt_scores = day_score[tgt_days - 1]

    # Quantile thresholds over target days
    storm_thr = np.quantile(tgt_scores, storm_top_q)
    calm_thr  = np.quantile(tgt_scores, calm_bottom_q)

    storm_days = set(tgt_days[tgt_scores >= storm_thr].tolist())
    calm_pool  = tgt_days[tgt_scores <= calm_thr].tolist()
    if len(calm_pool) == 0:
        print("[EventEval] No calm days found after anchor; skipping calm-heavy metric.")
        calm_days = set()
    else:
        k = int(min(calm_n, len(calm_pool)))
        calm_days = set(rng.choice(calm_pool, size=k, replace=False).tolist())

    # Map selected days -> timestamps (ensure datetime64[ns])
    ts_storm = _timestamps_for_days(ts, storm_days)
    ts_calm  = _timestamps_for_days(ts, calm_days)

    # Filter both frames to the selected timestamps and compute RMSE
    if len(ts_calm) > 0:
        truth_calm, pred_calm = _filter_truth_pred_by_days(df_truth_val, df_pred_val, ts_calm)
        rmse_calm = per_county_rmse_mean(truth_calm, pred_calm)
        print(f"[EventEval] Calm-heavy ({len(calm_days)} days): RMSE={rmse_calm:.4f}")
    else:
        print("[EventEval] Calm-heavy: RMSE=nan (no calm days)")

    if len(ts_storm) > 0:
        truth_storm, pred_storm = _filter_truth_pred_by_days(df_truth_val, df_pred_val, ts_storm)
        rmse_storm = per_county_rmse_mean(truth_storm, pred_storm)
        print(f"[EventEval] Storm-heavy ({len(storm_days)} days): RMSE={rmse_storm:.4f}")
    else:
        print("[EventEval] Storm-heavy: RMSE=nan (no storm days)")
