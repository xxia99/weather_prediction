import os
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from demo_evaluate import load_truth_wide, per_county_rmse_mean

# ======================== config ========================
SPLIT_DIR = "data/"
OUT_DIR   = "results_seq_tcn_minimal_keepval"
os.makedirs(OUT_DIR, exist_ok=True)

# core training
SEQ_LEN       = 24
BATCH_SIZE    = 64
EPOCHS        = 40
LR            = 8e-4
WEIGHT_DECAY  = 1e-4
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"

# model width/depth
TCN_D_MODEL   = 64
TCN_DROPOUT   = 0.15
TCN_KSIZE     = 3
TCN_LEVELS    = 4            # dilations: 1,2,4,8

# quantiles + point
USE_QUANTILES  = True
QUANTILES      = [0.1, 0.5, 0.9]
PRED_QUANTILE  = 0.90
HUBER_DELTA    = 1.0         # for point aux loss
POINT_BLEND    = np.array([0.2, 0.35, 0.45], dtype=np.float32)  # to align with RMSE

# learning tricks
USE_SWA             = True   # Stochastic Weight Averaging for final 25% epochs
SWA_START_FRAC      = 0.75
MAX_GRAD_NORM       = 5.0

# day split + validation strategy (unchanged)
TRAIN_LAST_DAY = 80
EVENT_METHOD   = "mean"
STORM_TOP_Q    = 0.80
CALM_BOTTOM_Q  = 0.50
CALM_N         = 10
BEST_BY        = "overall"

ZERO_EPS = 1e-8

# =======================================================
# utils
# =======================================================
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
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return np.empty((0, seq_len, X_loc.shape[1]), dtype=float), np.empty((0, horizon), dtype=float)
    Xw, Yw = [], []
    for i in range(N):
        Xw.append(X_loc[i:i+seq_len])
        Yw.append(y_loc[i+seq_len:i+seq_len+horizon])
    return np.asarray(Xw, dtype=float), np.asarray(Yw, dtype=float)

# =======================================================
# TCN forecaster (minimal, regularized)
# =======================================================
def _causal_conv1d(x, conv, dilation, ksize):
    # x: (B*N, C, T)
    pad = (ksize - 1) * dilation
    x = nn.functional.pad(x, (pad, 0))
    return conv(x)

class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, ksize=3, dilation=1, dropout=0.1):
        super().__init__()
        self.ksize = ksize
        self.dil = dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=ksize, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=ksize, dilation=dilation)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.res   = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.norm1 = nn.BatchNorm1d(c_out)
        self.norm2 = nn.BatchNorm1d(c_out)

    def forward(self, x):
        # x: (B*N, C, T)
        y = _causal_conv1d(x, self.conv1, self.dil, self.ksize)
        y = self.norm1(self.act(y))
        y = self.drop(y)
        y = _causal_conv1d(y, self.conv2, self.dil, self.ksize)
        y = self.norm2(self.act(y))
        y = self.drop(y)
        return y + self.res(x)

class TCNForecaster(nn.Module):
    """
    Input:  (B, T, N, F)
    Output: (B, N, H, Q) if USE_QUANTILES else (B, N, H)
    """
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels
        self.n_quantiles = len(QUANTILES) if USE_QUANTILES else 1
        out_dim = horizon * self.n_quantiles

        self.in_proj = nn.Linear(in_channels, TCN_D_MODEL)

        layers = []
        c = TCN_D_MODEL
        for i in range(TCN_LEVELS):
            dil = 2 ** i
            layers.append(_TCNBlock(c, c, ksize=TCN_KSIZE, dilation=dil, dropout=TCN_DROPOUT))
        self.tcn = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(TCN_D_MODEL, TCN_D_MODEL),
            nn.GELU(),
            nn.Dropout(TCN_DROPOUT),
            nn.Linear(TCN_D_MODEL, out_dim),
        )

    def forward(self, x):
        B, T, N, F = x.shape
        assert N == self.num_nodes and F == self.in_channels, \
            f"TCN input mismatch: got N={N},F={F}, expected N={self.num_nodes},F={self.in_channels}"

        # (B,T,N,F) -> (B*N, T, F) -> project -> (B*N, C, T)
        x = x.permute(0,2,1,3).contiguous().view(B*N, T, F)
        x = self.in_proj(x)                     # (B*N, T, C)
        x = x.permute(0,2,1).contiguous()      # (B*N, C, T)

        h = self.tcn(x)                         # (B*N, C, T)
        h_last = h[:, :, -1]                    # (B*N, C)
        y = self.head(h_last)                   # (B*N, H*Q or H)
        y = y.view(B, N, self.horizon, self.n_quantiles)
        return y

# =======================================================
# day split & scalers
# =======================================================
def _compute_day_indices(ts: np.ndarray):
    ts_pd = pd.to_datetime(ts)
    days_norm = ts_pd.normalize()
    uniq = days_norm.unique()
    uniq = pd.DatetimeIndex(sorted(uniq))
    day_map = {d: i+1 for i, d in enumerate(uniq)}
    days = np.array([day_map[d] for d in days_norm], dtype=int)
    day_end_idx = []
    for d in uniq:
        mask = (days_norm == d)
        idxs = np.where(mask)[0]
        day_end_idx.append(int(idxs.max()))
    return days, list(uniq), day_end_idx

def _fit_scalers_train_only(y, w, days, train_last_day):
    mask = days <= train_last_day
    y_tr = y[mask, :].reshape(-1, 1)
    w_tr = w[mask, :, :].reshape(-1, w.shape[2])
    y_mu, y_sd = zfit(y_tr)
    w_mu, w_sd = zfit(w_tr)
    return {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}

# =======================================================
# training split
# =======================================================
def prepare_training_split(ds_train, horizon, train_last_day=TRAIN_LAST_DAY):
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, uniq_days, day_end_idx = _compute_day_indices(ts)
    T, N, F = w.shape
    Din = 1 + F

    scalers = _fit_scalers_train_only(y, w, days, train_last_day)
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)   # (T, N, Din)

    day_last_idx = day_end_idx[train_last_day - 1]
    N_train = max(0, day_last_idx - SEQ_LEN - horizon + 1)

    X_list, Y_list = [], []
    for node in range(N):
        x_loc_full = X_all[:, node, :]
        Xw, Yw = build_windows(x_loc_full, y_sc[:, node], SEQ_LEN, horizon)
        if len(Xw):
            X_list.append(Xw[:N_train][:, :, None, :])  # (nW, SEQ_LEN, 1, Din)
            Y_list.append(Yw[:N_train][:, None, :])     # (nW, 1, H)

    if not X_list:
        raise RuntimeError("No training windows after split; check SEQ_LEN/horizon/TRAIN_LAST_DAY.")
    X_tr = np.concatenate(X_list, axis=0)
    Y_tr = np.concatenate(Y_list, axis=0)
    aux = {
        "X_all_scaled": X_all,
        "y_scaled": y_sc,
        "timestamps": ts,
        "days": days,
        "day_end_idx": day_end_idx
    }
    return X_tr, Y_tr, Din, scalers, aux

# =======================================================
# validation roll (full pipeline preserved)
# =======================================================
@torch.no_grad()
def predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day=TRAIN_LAST_DAY):
    model.eval()
    X_all = aux["X_all_scaled"]
    ts    = aux["timestamps"]
    days  = aux["days"]
    day_end_idx = aux["day_end_idx"]
    N = X_all.shape[1]
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
        for i, county in enumerate(locs):
            Xin_i = X_all[start_ctx:end_idx+1, i, :].reshape(1, SEQ_LEN, 1, X_all.shape[2])
            pred_sc = infer(model, Xin_i)[0, 0]  # (H,)
            y_mu = float(scalers["y_mu"].flatten()[0])
            y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0
            pred = (pred_sc * y_sd) + y_mu
            pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)
            rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))

    df_pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp", "location", "pred"])
    return df_pred

def truth_wide_for_val(ds_train, aux, horizon, train_last_day=TRAIN_LAST_DAY):
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

    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    locs = list(map(str, ds_train.location.values))
    df = pd.DataFrame(y[fut_indices, :], columns=locs, index=ts_fut).reset_index().rename(columns={"index": "timestamp"})
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

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

def _zero_nonzero_ratios(df_truth_wide: pd.DataFrame, df_pred_long: pd.DataFrame, eps: float = ZERO_EPS):
    if df_truth_wide is None or df_truth_wide.empty or df_pred_long is None or df_pred_long.empty:
        return float("nan"), float("nan")

    df_t = df_truth_wide.copy()
    df_t["timestamp"] = pd.to_datetime(df_t["timestamp"], errors="coerce").astype("datetime64[ns]")
    df_t = df_t.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    tw = df_t.set_index("timestamp").sort_index()

    df_p = df_pred_long.copy()
    df_p["timestamp"] = pd.to_datetime(df_p["timestamp"], errors="coerce").astype("datetime64[ns]")
    df_p["location"]  = df_p["location"].astype(str)
    df_p = df_p.dropna(subset=["timestamp", "location"])
    df_p = df_p.groupby(["timestamp", "location"], as_index=False, sort=False)["pred"].mean()
    pw = df_p.pivot_table(index="timestamp", columns="location", values="pred", aggfunc="mean").sort_index()

    common_ts = tw.index.intersection(pw.index)
    if len(common_ts) == 0:
        return float("nan"), float("nan")
    tw = tw.loc[common_ts]
    pw = pw.loc[common_ts]

    locs = sorted(set(tw.columns).intersection(pw.columns))
    if len(locs) == 0:
        return float("nan"), float("nan")

    y_true = tw[locs].to_numpy()
    y_pred = pw[locs].to_numpy()

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan"), float("nan")

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    true_zero = (np.abs(y_true) <= eps)
    pred_zero = (np.abs(y_pred) <= eps)
    denom_zero = np.sum(true_zero)
    z_to_nz = np.sum(true_zero & (~pred_zero)) / denom_zero if denom_zero > 0 else float("nan")
    denom_nz = np.sum(~true_zero)
    nz_to_z = np.sum((~true_zero) & pred_zero) / denom_nz if denom_nz > 0 else float("nan")
    return float(z_to_nz), float(nz_to_z)

# -------- event helpers & callback (unchanged) --------
def _per_day_outage_score(ds_train, method: str = "mean"):
    ts = ds_train.timestamp.values
    days, _, _ = _compute_day_indices(ts)
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
    days, _, _ = _compute_day_indices(ts_pd)
    mask = np.array([ (d in select_days) for d in days ], dtype=bool)
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

def _event_rmse_blocks(ds_train, df_truth_val, df_pred_val,
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

def make_val_callback_rolling(ds_train, scalers, horizon, aux, label,
                              train_last_day=TRAIN_LAST_DAY,
                              ts_future_test=None, df_truth_test=None, zero_rmse_test=float("nan")):
    best = {"rmse": float("inf"), "epoch": -1}
    df_truth_val = truth_wide_for_val(ds_train, aux, horizon, train_last_day)

    def _callback(model, epoch):
        df_pred_val = predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day)

        (rmse_overall, zero_overall,
         rmse_calm, zero_calm, n_calm,
         rmse_storm, zero_storm, n_storm) = _event_rmse_blocks(
            ds_train, df_truth_val, df_pred_val,
            train_last_day=train_last_day,
            method=EVENT_METHOD,
            storm_top_q=STORM_TOP_Q,
            calm_bottom_q=CALM_BOTTOM_Q,
            calm_n=CALM_N,
            seed=42
        )

        zpnz_model, nzpz_model = _zero_nonzero_ratios(df_truth_val, df_pred_val, eps=ZERO_EPS)
        if not df_truth_val.empty:
            locs = [c for c in df_truth_val.columns if c != "timestamp"]
            df_zero_long = pd.DataFrame({
                "timestamp": np.repeat(df_truth_val["timestamp"].values, len(locs)),
                "location": np.tile(locs, len(df_truth_val)),
                "pred": 0.0
            })
            zpnz_zero, nzpz_zero = _zero_nonzero_ratios(df_truth_val, df_zero_long, eps=ZERO_EPS)
        else:
            zpnz_zero, nzpz_zero = float("nan"), float("nan")

        if np.isfinite(rmse_overall):
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE={rmse_overall:.4f} (all-zero={zero_overall:.4f})")
        else:
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE=nan (all-zero=nan)")

        print(f"[Val {label}] Epoch {epoch}: Zero→Nonzero={zpnz_model if np.isfinite(zpnz_model) else float('nan'):.4f} "
              f"(all-zero={zpnz_zero if np.isfinite(zpnz_zero) else float('nan'):.4f}) | "
              f"Nonzero→Zero={nzpz_model if np.isfinite(nzpz_model) else float('nan'):.4f} "
              f"(all-zero={nzpz_zero if np.isfinite(nzpz_zero) else float('nan'):.4f})")

        if np.isfinite(rmse_calm):
            print(f"[EventEval {label}] Calm-heavy ({n_calm} days): RMSE={rmse_calm:.4f} (all-zero={zero_calm:.4f})")
        else:
            print(f"[EventEval {label}] Calm-heavy: RMSE=nan (all-zero=nan)")

        if np.isfinite(rmse_storm):
            print(f"[EventEval {label}] Storm-heavy ({n_storm} days): RMSE={rmse_storm:.4f} (all-zero={zero_storm:.4f})")
        else:
            print(f"[EventEval {label}] Storm-heavy: RMSE=nan (all-zero=nan)")

        metric_map = {"overall": rmse_overall, "calm": rmse_calm, "storm": rmse_storm}
        sel = metric_map.get(str(BEST_BY).lower(), rmse_overall)
        sel_val = sel if np.isfinite(sel) else rmse_overall

        if np.isfinite(sel_val) and sel_val < best["rmse"]:
            best["rmse"] = sel_val
            best["epoch"] = epoch
            save_path = os.path.join(OUT_DIR, f"best_enc-dec_{label}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "label": label,
                    "rmse_select": sel_val,
                    "rmse_overall": rmse_overall,
                    "rmse_calm": rmse_calm,
                    "rmse_storm": rmse_storm,
                    "best_by": str(BEST_BY).lower(),
                    "model_state_dict": model.state_dict(),
                },
                save_path,
            )
            print(f"[Val {label}] New best ({BEST_BY}) RMSE {sel_val:.4f} at epoch {epoch}. Saved: {save_path}")

            ts_map = pd.DataFrame({"timestamp": pd.to_datetime(aux["timestamps"]), "day": aux["days"]})
            df_pred_day  = df_pred_val.merge(ts_map, on="timestamp", how="left")
            df_truth_day = df_truth_val.merge(ts_map, on="timestamp", how="left")

            unique_days = sorted(df_pred_day["day"].dropna().unique())
            for d in unique_days:
                pred_d  = df_pred_day[df_pred_day["day"] == d].drop(columns="day")
                truth_d = df_truth_day[df_truth_day["day"] == d].drop(columns="day")
                pred_path  = os.path.join(OUT_DIR, f"val_pred_epoch{epoch}_day{int(d)}_{label}.csv")
                truth_path = os.path.join(OUT_DIR, f"val_truth_epoch{epoch}_day{int(d)}_{label}.csv")
                pred_d.to_csv(pred_path, index=False); truth_d.to_csv(truth_path, index=False)
                print(f"[Val {label}] Saved day {int(d)} prediction/truth data: {pred_path}, {truth_path}")

            if unique_days:
                fig, axes = plt.subplots(len(unique_days), 1, figsize=(8, 3 * len(unique_days)), sharex=False)
                if len(unique_days) == 1: axes = [axes]
                for ax, d in zip(axes, unique_days):
                    truth_d = df_truth_day[df_truth_day["day"] == d].set_index("timestamp").drop(columns="day")
                    truth_mean = truth_d.mean(axis=1).reset_index(name="truth")
                    pred_mean  = df_pred_day[df_pred_day["day"] == d].groupby("timestamp")["pred"].mean().reset_index()
                    merged = pd.merge(truth_mean, pred_mean, on="timestamp", how="inner").sort_values("timestamp")
                    ax.plot(merged["timestamp"], merged["truth"], label="Truth")
                    ax.plot(merged["timestamp"], merged["pred"], label="Prediction")
                    ax.set_title(f"Day {int(d)}"); ax.set_xlabel("timestamp"); ax.set_ylabel("outage"); ax.legend()
                fig.tight_layout()
                fig_path = os.path.join(OUT_DIR, f"val_curve_epoch{epoch}_{label}.png")
                fig.savefig(fig_path)
                plt.close(fig)
                print(f"[Val {label}] Saved prediction vs truth plot: {fig_path}")

        if ts_future_test is not None and df_truth_test is not None and len(ts_future_test) > 0:
            df_pred_test = predict_per_county(ds_train, model, scalers, horizon, ts_future_test)
            rmse_test = per_county_rmse_mean(df_truth_test, df_pred_test)

            zpnz_t, nzpz_t = _zero_nonzero_ratios(df_truth_test, df_pred_test, eps=ZERO_EPS)
            locs_t = [c for c in df_truth_test.columns if c != "timestamp"]
            df_zero_long_t = pd.DataFrame({
                "timestamp": np.repeat(df_truth_test["timestamp"].values, len(locs_t)),
                "location": np.tile(locs_t, len(df_truth_test)),
                "pred": 0.0
            })
            zpnz_t0, nzpz_t0 = _zero_nonzero_ratios(df_truth_test, df_zero_long_t, eps=ZERO_EPS)

            print(f"[Val {label}] Epoch {epoch}: Testing RMSE={rmse_test:.4f} (all-zero RMSE={zero_rmse_test:.4f})")
            print(f"[Val {label}] Epoch {epoch}: Test Zero→Nonzero={zpnz_t if np.isfinite(zpnz_t) else float('nan'):.4f} "
                  f"(all-zero={zpnz_t0 if np.isfinite(zpnz_t0) else float('nan'):.4f}) | "
                  f"Nonzero→Zero={nzpz_t if np.isfinite(nzpz_t) else float('nan'):.4f} "
                  f"(all-zero={nzpz_t0 if np.isfinite(nzpz_t0) else float('nan'):.4f})")

    return _callback

# =======================================================
# training / inference
# =======================================================
def _pinball_loss(yq, y_true, q_list, quantile_weights=None):
    # yq: (B,N,H,Q), y_true: (B,N,H)
    qs = torch.tensor(q_list, device=yq.device, dtype=yq.dtype).view(1,1,1,-1)
    diff = y_true.unsqueeze(-1) - yq
    loss = torch.maximum(qs * diff, (qs - 1.0) * diff)    # pinball
    if quantile_weights is not None:
        qw = torch.tensor(quantile_weights, device=yq.device, dtype=yq.dtype)
        qw = qw / torch.clamp(qw.sum(), min=1e-8)
        loss = loss * qw.view(1,1,1,-1)
    return loss.mean()

def _huber(x, delta=1.0):
    absx = torch.abs(x)
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device, dtype=x.dtype))
    lin  = absx - quad
    return 0.5 * quad * quad + delta * lin

def train_model(X, Y, din, horizon, n_nodes, eval_callback=None):
    if len(X) == 0:
        return None

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = TCNForecaster(in_channels=din, horizon=horizon, num_nodes=n_nodes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ReduceLROnPlateau on training loss
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    # Optional SWA in the tail
    if USE_SWA:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_start = int(EPOCHS * SWA_START_FRAC)
        swa_model = AveragedModel(model)
        swa_sched = SWALR(opt, swa_lr=LR * 0.5)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            yq = model(xb)                # (B,N,H,Q)

            # pinball + small huber on blended point to align with RMSE
            loss = _pinball_loss(yq, yb, QUANTILES, quantile_weights=POINT_BLEND)
            with torch.no_grad():
                qw = torch.tensor(POINT_BLEND, device=yq.device, dtype=yq.dtype)
                qw = qw / qw.sum()
            y_point = (yq * qw.view(1,1,1,-1)).sum(-1)    # (B,N,H)
            loss = loss + 0.15 * _huber(y_point - yb, delta=HUBER_DELTA).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        # scheduler step
        sched.step(epoch_loss)

        # SWA averaging after swa_start
        if USE_SWA and (epoch >= swa_start):
            swa_model.update_parameters(model)
            swa_sched.step()
        print(f"[Epoch {epoch+1}/{EPOCHS}] TrainLoss={epoch_loss:.6f} | LR={opt.param_groups[0]['lr']:.2e}")

        if eval_callback is not None:
            eval_callback(model if not USE_SWA or epoch < swa_start else swa_model, epoch + 1)

    # If SWA used, update BN stats and return swa weights
    if USE_SWA:
        from torch.optim.swa_utils import update_bn
        update_bn(dl, swa_model)
        return swa_model
    return model

@torch.no_grad()
def infer(model, Xin):
    model.eval()
    yq = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE))  # (1,N,H,Q)
    if USE_QUANTILES:
        qs = np.array(QUANTILES, dtype=float)
        idx = int(np.argmin(np.abs(qs - float(PRED_QUANTILE))))
        out = yq[..., idx]
    else:
        out = yq
    return out.cpu().numpy()

# =======================================================
# test-time prediction (unchanged interface)
# =======================================================
def predict_per_county(ds_train, model, scalers, horizon, ts_future):
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float)
    T, N, F = w.shape
    Din = 1 + F

    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)

    rows = []
    for i, county in enumerate(locs):
        Xin_i = X_all[-SEQ_LEN:, i, :].reshape(1, SEQ_LEN, 1, Din)
        pred_sc = infer(model, Xin_i)[0, 0]

        pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)
        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))
    return pd.concat(rows, ignore_index=True)

# =======================================================
# main (pipeline preserved: plots, csvs, metrics)
# =======================================================
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))

    _ = get_all_features(ds_train)
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))

    zero_rmse_24 = _all_zero_rmse_for_truth(df_t24)
    zero_rmse_48 = _all_zero_rmse_for_truth(df_t48)

    # ---- Prepare once
    X24, Y24, Din, scalers24, aux24 = prepare_training_split(ds_train, horizon=H24, train_last_day=TRAIN_LAST_DAY)
    X48, Y48, _,  scalers48, aux48  = prepare_training_split(ds_train, horizon=H48, train_last_day=TRAIN_LAST_DAY)
    n_nodes = 1

    # ---- 24h
    val_cb24 = make_val_callback_rolling(ds_train, scalers24, H24, aux24, label="24h",
                                         train_last_day=TRAIN_LAST_DAY,
                                         ts_future_test=ts24, df_truth_test=df_t24, zero_rmse_test=zero_rmse_24)
    mdl24 = train_model(X24, Y24, Din, H24, n_nodes, eval_callback=val_cb24)

    # ---- 48h
    val_cb48 = make_val_callback_rolling(ds_train, scalers48, H48, aux48, label="48h",
                                         train_last_day=TRAIN_LAST_DAY,
                                         ts_future_test=ts48, df_truth_test=df_t48, zero_rmse_test=zero_rmse_48)
    mdl48 = train_model(X48, Y48, Din, H48, n_nodes, eval_callback=val_cb48)

    # ----- Final test with best checkpoint if present -----
    def _finalize(model, scalers, H, ts_future, df_truth, tag, zero_rmse):
        best_path = os.path.join(OUT_DIR, f"best_enc-dec_{tag}.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=DEVICE)
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"[{tag}] Loaded best checkpoint (epoch {ckpt.get('epoch')}, val-selected RMSE={ckpt.get('rmse_select', ckpt.get('rmse', float('nan'))):.4f})")
            except Exception as e:
                print(f"[{tag}] Warning: failed to load best checkpoint: {e}")
        df_pred = predict_per_county(ds_train, model, scalers, H, ts_future)
        rmse = per_county_rmse_mean(df_truth, df_pred)
        print(f"[Test {tag}] RMSE={rmse:.4f} (all-zero RMSE={zero_rmse:.4f})")

        # diagnostics
        zpnz, nzpz = _zero_nonzero_ratios(df_truth, df_pred, eps=ZERO_EPS)
        locs = [c for c in df_truth.columns if c != "timestamp"]
        df_zero_long = pd.DataFrame({"timestamp": np.repeat(df_truth['timestamp'].values, len(locs)),
                                     "location": np.tile(locs, len(df_truth)),
                                     "pred": 0.0})
        zpnz0, nzpz0 = _zero_nonzero_ratios(df_truth, df_zero_long, eps=ZERO_EPS)
        print(f"[Test {tag}] Zero→Nonzero={zpnz if np.isfinite(zpnz) else float('nan'):.4f} (all-zero={zpnz0 if np.isfinite(zpnz0) else float('nan'):.4f}) | "
              f"Nonzero→Zero={nzpz if np.isfinite(nzpz) else float('nan'):.4f} (all-zero={nzpz0 if np.isfinite(nzpz0) else float('nan'):.4f})")
        return df_pred

    df24 = _finalize(mdl24, scalers24, H24, ts24, df_t24, "24h", zero_rmse_24)
    df48 = _finalize(mdl48, scalers48, H48, ts48, df_t48, "48h", zero_rmse_48)

    out24 = os.path.join(OUT_DIR, "stemgnn_pred_24h.csv")
    out48 = os.path.join(OUT_DIR, "stemgnn_pred_48h.csv")
    df24.to_csv(out24, index=False); df48.to_csv(out48, index=False)
    print("Saved predictions:\n  ", out24, "\n  ", out48)

    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
