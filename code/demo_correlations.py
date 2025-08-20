import os
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

SPLIT_DIR = "data/"
OUT_DIR = "results_vis/"
os.makedirs(OUT_DIR, exist_ok=True)


def open_ds(path: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path)


def plot_spatial_corr(df: pd.DataFrame, title: str, out_path: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_temporal_autocorr(series: pd.Series, max_lag: int, title: str, out_path: str) -> None:
    ac = [series.autocorr(lag) for lag in range(1, max_lag + 1)]
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, max_lag + 1), ac)
    plt.xlabel("Lag (hours)")
    plt.ylabel("Autocorrelation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    ds = open_ds(os.path.join(SPLIT_DIR, "train.nc"))

    out_df = ds.out.transpose("timestamp", "location").to_pandas()
    plot_spatial_corr(
        out_df,
        "Spatial correlation of outage counts",
        os.path.join(OUT_DIR, "spatial_outage.png"),
    )
    plot_temporal_autocorr(
        pd.Series(out_df.values.reshape(-1)),
        48,
        "Temporal autocorrelation of outage counts",
        os.path.join(OUT_DIR, "temporal_outage.png"),
    )

    if "t2m" in ds.weather.feature.values:
        t2m_df = (
            ds.weather.sel(feature="t2m").transpose("timestamp", "location").to_pandas()
        )
        plot_spatial_corr(
            t2m_df,
            "Spatial correlation of 2m temperature",
            os.path.join(OUT_DIR, "spatial_t2m.png"),
        )
        plot_temporal_autocorr(
            pd.Series(t2m_df.values.reshape(-1)),
            48,
            "Temporal autocorrelation of 2m temperature",
            os.path.join(OUT_DIR, "temporal_t2m.png"),
        )

    ds.close()
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
