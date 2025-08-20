# train_simple.py
# Trains a simple outages-only model per county (SARIMAX(1,0,1)),
# then writes per-county predictions for both horizons as CSVs.

import os
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.statespace.sarimax import SARIMAX

SPLIT_DIR = "data/"
OUT_DIR = "results/"
os.makedirs(OUT_DIR, exist_ok=True)

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

def main():
    ds_train = xr.open_dataset(os.path.join(SPLIT_DIR, "train.nc"))
    # ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h.nc"))
    # ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h.nc"))
    ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))

    counties = list(ds_train.location.values)
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)

    rows24 = []
    rows48 = []

    for county in counties:
        y_train = ds_train.out.sel(location=county).values.astype(float).flatten()

        res = safe_fit_sarimax(y_train)

        # Forecast horizons
        if res is None:
            pred24 = np.zeros(len(ts24), dtype=float)
            pred48 = np.zeros(len(ts48), dtype=float)
        else:
            pred48 = np.asarray(res.forecast(steps=len(ts48)), dtype=float)
            pred24 = pred48[:len(ts24)]

        # Collect rows
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

    df24.to_csv(os.path.join(OUT_DIR, "sarimax_pred_24h.csv"), index=False)
    df48.to_csv(os.path.join(OUT_DIR, "sarimax_pred_48h.csv"), index=False)

    print("Saved predictions:")
    print(f"  {os.path.join(OUT_DIR, 'sarimax_pred_24h.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'sarimax_pred_48h.csv')}")

    # Clean up
    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
