# rank_surges.py
"""
Rank daily outage surges and label their severity levels.

Usage
-----
python rank_surges.py \
  --data data/train.nc \
  --outdir results_surge \
  --agg mean \
  --first_n_days 90

--agg: mean (robust) or max (worst-county focus)
--first_n_days: set to 90 for first 90 days; use 'none' to use all days.
"""

import os
import argparse
import json
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


def open_ds(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path, engine="netcdf4")


def compute_day_indices(ts_np):
    """Return per-timestamp day index (1..K), unique normalized dates, and day end indices."""
    ts = pd.to_datetime(ts_np)
    days_norm = ts.normalize()
    uniq = pd.DatetimeIndex(sorted(days_norm.unique()))
    day_map = {d: i + 1 for i, d in enumerate(uniq)}
    days = np.array([day_map[d] for d in days_norm], dtype=int)
    day_end_idx = []
    for d in uniq:
        idxs = np.where(days_norm == d)[0]
        day_end_idx.append(int(idxs.max()))
    return days, uniq, day_end_idx


def zfit(arr):
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    return mu, (1.0 if sd == 0 else sd)


def make_levels(scores: np.ndarray):
    """Return thresholds and a function that maps score -> level name."""
    q25, q50, q75, q90 = np.nanquantile(scores, [0.25, 0.50, 0.75, 0.90]).tolist()

    def label(x):
        if x >= q90:
            return "Extreme"
        elif x >= q75:
            return "Severe"
        elif x >= q50:
            return "Moderate"
        elif x >= q25:
            return "Mild"
        else:
            return "Calm"

    thr = {"q25": q25, "q50": q50, "q75": q75, "q90": q90}
    return thr, label


def rank_surges(
    ds_path: str,
    outdir: str,
    agg: str = "mean",
    first_n_days: Optional[int] = 90,
):
    os.makedirs(outdir, exist_ok=True)
    ds = open_ds(ds_path)

    # Pull arrays (T,N), timestamps
    y = ds["out"].transpose("timestamp", "location").values.astype(float)  # (T, N)
    ts = pd.to_datetime(ds["timestamp"].values)

    # Day indices and optional restriction to first N days
    days, uniq_dates, _ = compute_day_indices(ts)
    if first_n_days is not None:
        keep_days = set(range(1, min(first_n_days, int(days.max())) + 1))
        mask = np.array([d in keep_days for d in days], dtype=bool)
        y = y[mask, :]
        ts = ts[mask]
        days, uniq_dates, _ = compute_day_indices(ts)

    # Aggregate across locations per hour
    agg = str(agg).lower()
    if agg == "mean":
        per_hour = np.nanmean(y, axis=1)  # (T,)
    elif agg == "max":
        per_hour = np.nanmax(y, axis=1)
    else:
        raise ValueError("--agg must be 'mean' or 'max'")

    # Build per-day metrics
    records = []
    for i, day_date in enumerate(uniq_dates, start=1):
        mask = days == i
        x = per_hour[mask]  # per-hour aggregated series for day i
        if x.size == 0:
            continue
        day_mean = float(np.nanmean(x))
        day_max = float(np.nanmax(x))
        day_p90 = float(np.nanquantile(x, 0.90))
        spike_idx = float(day_max - day_mean)
        records.append(
            {
                "day_index": i,
                "date": pd.Timestamp(day_date).date(),
                "day_mean": day_mean,
                "day_max": day_max,
                "day_p90": day_p90,
                "spike_idx": spike_idx,
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No days found after filtering; check your inputs.")

    # Standardize metrics and compute a composite surge score
    for col in ["day_max", "day_p90", "spike_idx"]:
        mu, sd = zfit(df[col].values)
        df[f"z_{col}"] = (df[col] - mu) / sd

    # Composite: emphasize max (shape), keep p90 (robust tail), include spike index (contrast)
    df["surge_score"] = 0.5 * df["z_day_max"] + 0.3 * df["z_day_p90"] + 0.2 * df["z_spike_idx"]

    # Label severity levels by score quantiles
    thr, lab_fn = make_levels(df["surge_score"].values)
    df["level"] = df["surge_score"].map(lab_fn)

    # Rank high → low
    df = df.sort_values(["surge_score", "day_max", "day_p90"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    # Save
    out_csv = os.path.join(outdir, f"surge_ranking_{agg}.csv")
    df.to_csv(out_csv, index=False)

    thr_path = os.path.join(outdir, f"surge_thresholds_{agg}.json")
    with open(thr_path, "w") as f:
        json.dump(thr, f, indent=2, default=float)

    # Console report
    print(f"[Surge Ranking] Aggregation: {agg.upper()} | Days: {len(df)}")
    print(f"[Surge Ranking] Saved: {out_csv}")
    print(f"[Surge Ranking] Level thresholds (by surge_score quantiles): {thr}")
    print("\nTop 10 surge days:")
    print(df.loc[:9, ["rank", "day_index", "date", "level", "surge_score", "day_max", "day_p90", "spike_idx"]].to_string(index=False))

    ds.close()
    return df


def _maybe_int_or_none(text: str) -> Optional[int]:
    t = str(text).strip().lower()
    if t in ("none", "all", ""):
        return None
    return int(text)


def main():
    ap = argparse.ArgumentParser(description="Rank daily outage surges and label severity levels.")
    ap.add_argument("--data", type=str, default="data/train.nc", help="Path to NetCDF with variables 'out' and 'timestamp'.")
    ap.add_argument("--outdir", type=str, default="results_surge", help="Output directory for CSV results.")
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "max"], help="Per-hour aggregation across locations.")
    ap.add_argument("--first_n_days", type=_maybe_int_or_none, default=90, help="Restrict to first N days; use 'none' for all.")
    args = ap.parse_args()

    rank_surges(ds_path=args.data, outdir=args.outdir, agg=args.agg, first_n_days=args.first_n_days)


if __name__ == "__main__":
    main()
