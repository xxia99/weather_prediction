import os
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from demo_evaluate import load_truth_wide, per_county_rmse_mean
from torch.optim.swa_utils import AveragedModel

def _load_ckpt_flex(model: nn.Module, sd: dict, strict: bool = True) -> bool:
    # 1) Try direct
    try:
        model.load_state_dict(sd, strict=strict)
        return True
    except Exception:
        pass

    # 2) If SWA wrapper and we were given a plain dict → load into .module
    if isinstance(model, AveragedModel):
        try:
            model.module.load_state_dict(sd, strict=strict)
            return True
        except Exception:
            pass

    # 3) Strip "module." prefix + drop "n_averaged" and try again
    if any(k.startswith("module.") for k in sd.keys()) or ("n_averaged" in sd):
        sd_stripped = {k.replace("module.", "", 1): v for k, v in sd.items() if k != "n_averaged"}
        try:
            # Load either into wrapper.module (if SWA) or directly (if plain)
            if isinstance(model, AveragedModel):
                model.module.load_state_dict(sd_stripped, strict=strict)
            else:
                model.load_state_dict(sd_stripped, strict=strict)
            return True
        except Exception:
            pass

    # 4) Final fallback: be permissive
    try:
        model.load_state_dict(sd, strict=False)
        return True
    except Exception:
        return False

# ======================== config ========================
SPLIT_DIR = "data/"

# core training
SEQ_LEN       = 24
BATCH_SIZE    = 64
EPOCHS        = 40
LR            = 8e-4
WEIGHT_DECAY  = 1e-4
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_GRAD_NORM = 5.0

# model width/depth
TCN_D_MODEL   = 64
TCN_DROPOUT   = 0.15
TCN_KSIZE     = 3
TCN_LEVELS    = 4            # dilations: 1,2,4,8

# --- County-specific MoE (minimal add) ---
NUM_EXPERTS     = 8          # small K; tune 4–8 if desired
COUNTY_EMB_DIM  = 16         # small embedding for counties

# heads / objectives
USE_ZINB_HEAD      = True         # added head: predicts (pi, mu, theta)
KEEP_QUANTILES     = True         # keep quantile head for RMSE alignment and diagnostics

# quantiles (kept for aux / inference if desired)
QUANTILES      = [0.1, 0.5, 0.9]
PRED_QUANTILE  = 0.90            # used if INFER_POINT="quantile"
HUBER_DELTA    = 1.0
POINT_BLEND    = np.array([0.2, 0.35, 0.45], dtype=np.float32)

# curriculum
PHASE1_EPOCHS      = 10           # 3–5 recommended
PRETRAIN_MODE      = "nb_mean"   # "nb_mean" or "quantile"
Q_HIGH             = [0.8, 0.9, 0.95]  # emphasized upper qs in phase 2 (if KEEP_QUANTILES)
LAMBDA_QHI         = 0.4         # weight for upper-quantile pinball in phase 2
LAMBDA_ZINB        = 1.0         # main ZINB NLL weight
LAMBDA_BCE_ZERO    = 0.05        # small aux BCE(pi vs zero indicator) in phase 1 (stabilizes π)

# surge-aware weighting (phase 2)
SURGE_PRIOR_DEFAULT = 0.20
SURGE_CSV_PATH      = "./results_surge/surge_ranking_mean.csv"
SURGE_LEVELS_STORM  = {"Severe", "Extreme"}
W_SURGE             = 4.0

# augmentation (small & realistic; applied only on surge windows by default)
AUG_ENABLE          = True
AUG_SURGE_ONLY      = True
AUG_PROB            = 0.5
JITTER_STD_FRAC     = 0.02
AMP_MAX_FRAC        = 0.10
WARP_MAX_FRAC       = 0.10
CUTMIX_PROB         = 0.30

# SWA
USE_SWA        = True
SWA_START_FRAC = 0.75

# day split + validation strategy (unchanged)
TRAIN_LAST_DAY = 85
EVENT_METHOD   = "mean"
STORM_TOP_Q    = 0.80
CALM_BOTTOM_Q  = 0.50
CALM_N         = 10
BEST_BY        = "calm"
# NEW: if True, override model selection to use RMSE from the LAST validation day only
SELECT_BEST_BY_LAST_VAL_DAY = True
ZERO_EPS = 1e-8

# inference choice
INFER_POINT = "zinb_mean"   # "zinb_mean" | "quantile"
SEED = 3

OUT_DIR   = f"results_MoE/per_county_{TRAIN_LAST_DAY}/{BEST_BY}_{NUM_EXPERTS}experts_fast/seed{SEED}/"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Reproducibility knobs ---

STRICT_DETERMINISM = True  # <- set True for bit-strict determinism (slower)
# =======================================================
# utils
# =======================================================
def _per_day_rmse(ds_train, df_truth_wide, df_pred_long, train_last_day):
    """
    Returns a list of dicts:
      [{"day": int, "rmse": float, "zero_rmse": float}], for all validation days > train_last_day.
    """
    if df_truth_wide.empty or df_pred_long.empty:
        return []

    ts_map = pd.DataFrame({"timestamp": pd.to_datetime(ds_train.timestamp.values),
                           "day": _compute_day_indices(ds_train.timestamp.values)[0]})
    # keep only timestamps present in both truth & pred
    ts_truth = pd.to_datetime(df_truth_wide["timestamp"].values)
    ts_pred  = pd.to_datetime(df_pred_long["timestamp"].values)
    ts_common = np.intersect1d(ts_truth.astype("datetime64[ns]"), ts_pred.astype("datetime64[ns]"))

    ts_day = ts_map[ts_map["timestamp"].isin(ts_common)]
    val_days = sorted([d for d in ts_day["day"].unique().tolist() if d > int(train_last_day)])

    rows = []
    for d in val_days:
        ts_subset = ts_day[ts_day["day"] == d]["timestamp"]
        truth_d, pred_d = _filter_truth_pred_by_days(df_truth_wide, df_pred_long, pd.DatetimeIndex(ts_subset))
        if not truth_d.empty:
            zero_d = _all_zero_rmse_for_truth(truth_d)
        else:
            zero_d = float("nan")
        rmse_d = per_county_rmse_mean(truth_d, pred_d) if (not truth_d.empty and not pred_d.empty) else float("nan")
        rows.append({
            "day": int(d),
            "rmse": float(rmse_d) if np.isfinite(rmse_d) else float("nan"),
            "zero_rmse": float(zero_d) if np.isfinite(zero_d) else float("nan")
        })
    return rows

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

# ---- Reproducibility helpers (updated for speed) ----------------------------
def set_seed(seed: int = 42, strict: bool = True):
    """
    Mirrors PyTorch's 'Controlling sources of randomness' recipe.
    Set strict=True for bitwise-repeatable runs (slower).
    """
    import os as _os, random as _random
    import numpy as _np
    import torch as _torch

    # 1) Python & NumPy
    _os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    _np.random.seed(seed)

    # 2) PyTorch CPU/GPU RNG
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)

    # 3) CuDNN / CuBLAS / algorithm determinism (strict per docs)
    #    NOTE: CuBLAS env var must be set before CUDA context is initialized.
    #    Calling it here is fine as long as you haven't done any CUDA work yet.
    # _torch.backends.cudnn.deterministic = True
    # _torch.backends.cudnn.benchmark = False

    # # TF32 can introduce tiny numeric diffs on Ampere+; disable for strict runs.
    # try:
    #     _torch.backends.cuda.matmul.allow_tf32 = False
    #     _torch.backends.cudnn.allow_tf32 = False
    # except Exception:
    #     pass

    # if strict:
    #     # Enforce deterministic algorithms and set CuBLAS workspace as per docs.
    #     # Prefer the larger workspace first; fall back if unavailable on your driver.
    #     if "CUBLAS_WORKSPACE_CONFIG" not in _os.environ:
    #         _os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #     try:
    #         _torch.use_deterministic_algorithms(True)
    #     except Exception:
    #         # older PyTorch versions
    #         pass
    # else:
    #     # Allow non-deterministic fast paths
    #     try:
    #         _torch.use_deterministic_algorithms(False)
    #     except Exception:
    #         pass

def seed_worker(worker_id):
    import torch as _torch, numpy as _np, random as _random
    worker_seed = _torch.initial_seed() % 2**32
    _np.random.seed(worker_seed)
    _random.seed(worker_seed)
# -----------------------------------------------------------------------------

def build_windows(X_loc, y_loc, seq_len, horizon):
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return np.empty((0, seq_len, X_loc.shape[1]), dtype=float), np.empty((0, horizon), dtype=float)
    Xw, Yw = [], []
    for i in range(N):
        Xw.append(X_loc[i:i+seq_len])
        Yw.append(y_loc[i+seq_len:i+seq_len+horizon])
    return np.asarray(Xw, dtype=float), np.asarray(Yw, dtype=float)

def _update_bn_on_device(loader, model, device):
    """Like torch.optim.swa_utils.update_bn, but moves inputs to the target device."""
    if not any(m.training for m in model.modules()):
        model.train()  # ensure BN runs in training mode

    momenta = {}
    model.apply(lambda module: momenta.setdefault(module, getattr(module, 'momentum', None))
                if isinstance(module, nn.modules.batchnorm._BatchNorm) else None)

    # Reset running stats
    for module in momenta.keys():
        module.running_mean.zero_()
        module.running_var.fill_(1)

    n = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        b = x.size(0)
        momentum = b / float(n + b)
        for module in momenta.keys():
            module.momentum = momentum
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            # --- pass a dummy county id of zeros for BN-only refresh ---
            if isinstance(x, (list, tuple)):
                model(x[0], torch.zeros(x[0].size(0), dtype=torch.long, device=device))
            else:
                model(x, torch.zeros(x.size(0), dtype=torch.long, device=device))
        n += b

    # restore original BN momenta
    for module, mom in momenta.items():
        module.momentum = mom


# =======================================================
# TCN forecaster (minimal backbone) + dual heads + county MoE
# =======================================================
def _causal_conv1d(x, conv, dilation, ksize):
    # x: (B*N, C, T)
    pad = (ksize - 1) * dilation
    x = F.pad(x, (pad, 0))
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

# --- minimal County-specific MoE bottleneck ---
class CountyMoE(nn.Module):
    def __init__(self, in_dim: int, num_counties: int, num_experts: int, emb_dim: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(num_counties, emb_dim)
        self.gate = nn.Linear(in_dim + emb_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim + emb_dim, in_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

    def forward(self, h_last: torch.Tensor, county_ids: torch.Tensor):
        """
        h_last: (B*N, C)
        county_ids: (B*N,) long
        """
        e = self.emb(county_ids)                           # (B*N, E)
        z = torch.cat([h_last, e], dim=-1)                # (B*N, C+E)
        logits = self.gate(z)                              # (B*N, K)
        gates  = torch.softmax(logits, dim=-1)            # (B*N, K)
        exp_outs = torch.stack([exp(z) for exp in self.experts], dim=1)  # (B*N, K, C)
        mixed = torch.sum(gates.unsqueeze(-1) * exp_outs, dim=1)         # (B*N, C)
        return mixed, gates

class TCNForecaster(nn.Module):
    """
    Input:  x  (B, T, N, F)  with N==1 in current pipeline
            county_ids (B,)   county index for each sample (from ds_train.location order)
    Output: dict {
        "q": (B, N, H, Q)  if KEEP_QUANTILES else None,
        "zinb": (logit_pi, raw_mu, raw_theta) each (B, N, H) if USE_ZINB_HEAD
    }
    """
    def __init__(self, in_channels, horizon, num_nodes, num_counties, num_experts, emb_dim):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels

        self.in_proj = nn.Linear(in_channels, TCN_D_MODEL)

        layers = []
        c = TCN_D_MODEL
        for i in range(TCN_LEVELS):
            dil = 2 ** i
            layers.append(_TCNBlock(c, c, ksize=TCN_KSIZE, dilation=dil, dropout=TCN_DROPOUT))
        self.tcn = nn.Sequential(*layers)

        # --- MoE bottleneck conditioned on county ---
        self.moe = CountyMoE(in_dim=TCN_D_MODEL,
                             num_counties=num_counties,
                             num_experts=num_experts,
                             emb_dim=emb_dim,
                             dropout=TCN_DROPOUT)

        # heads (unchanged), now consume MoE-mixed representation
        if KEEP_QUANTILES:
            out_q = horizon * len(QUANTILES)
            self.head_q = nn.Sequential(
                nn.Linear(TCN_D_MODEL, TCN_D_MODEL),
                nn.GELU(),
                nn.Dropout(TCN_DROPOUT),
                nn.Linear(TCN_D_MODEL, out_q),
            )
        else:
            self.head_q = None

        if USE_ZINB_HEAD:
            self.head_zinb = nn.Sequential(
                nn.Linear(TCN_D_MODEL, TCN_D_MODEL),
                nn.GELU(),
                nn.Dropout(TCN_DROPOUT),
                nn.Linear(TCN_D_MODEL, 3 * horizon),
            )
        else:
            self.head_zinb = None

    def forward(self, x, county_ids: torch.Tensor):
        B, T, N, F = x.shape
        assert N == self.num_nodes and F == self.in_channels, \
            f"TCN input mismatch: got N={N},F={F}, expected N={self.num_nodes},F={self.in_channels}"
        # (B,T,N,F) -> (B*N, T, F) -> project -> (B*N, C, T)
        x = x.permute(0,2,1,3).contiguous().view(B*N, T, F)
        x = self.in_proj(x)                     # (B*N, T, C)
        x = x.permute(0,2,1).contiguous()      # (B*N, C, T)

        h = self.tcn(x)                        # (B*N, C, T)
        h_last = h[:, :, -1]                   # (B*N, C)

        # county-specific MoE mix
        if county_ids is None:
            county_ids = torch.zeros(B*N, dtype=torch.long, device=h_last.device)
        else:
            if county_ids.dim() == 1 and county_ids.size(0) == B:
                county_ids = county_ids.view(B, 1).expand(B, N).reshape(B*N)
        h_mixed, _ = self.moe(h_last, county_ids)   # (B*N, C)

        out = {"q": None, "zinb": None}
        if self.head_q is not None:
            yq = self.head_q(h_mixed).view(B, N, self.horizon, len(QUANTILES))
            out["q"] = yq
        if self.head_zinb is not None:
            z = self.head_zinb(h_mixed).view(B, N, self.horizon, 3)
            logit_pi = z[..., 0]
            raw_mu   = z[..., 1]
            raw_th   = z[..., 2]
            out["zinb"] = (logit_pi, raw_mu, raw_th)
        return out

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

def _load_storm_prior(csv_path: str, train_last_day: int) -> float:
    if csv_path is None or (not os.path.exists(csv_path)):
        return float(SURGE_PRIOR_DEFAULT)
    try:
        df = pd.read_csv(csv_path)
        if "day_index" not in df.columns or "level" not in df.columns:
            return float(SURGE_PRIOR_DEFAULT)
        df = df[df["day_index"] <= int(train_last_day)].copy()
        total = max(1, len(df))
        storm = (df["level"].astype(str).isin(list(SURGE_LEVELS_STORM))).sum()
        prior = float(storm) / float(total)
        return float(np.clip(prior, 0.02, 0.8))
    except Exception:
        return float(SURGE_PRIOR_DEFAULT)

# =======================================================
# training split (also returns raw Y for ZINB and surge flags/weights)
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

    X_list, Y_sc_list, Y_raw_list, CID_list = [], [], [], []  # --- collect county ids ---
    for node in range(N):
        x_loc_full = X_all[:, node, :]
        Xw_sc, Yw_sc = build_windows(x_loc_full, y_sc[:, node], SEQ_LEN, horizon)
        # raw Y windows using *unscaled* y
        _, Yw_raw = build_windows(np.concatenate([y[:, node:node+1], w[:, node, :]], axis=1), y[:, node], SEQ_LEN, horizon)
        if len(Xw_sc):
            X_list.append(Xw_sc[:N_train][:, :, None, :])      # (nW, SEQ_LEN, 1, Din)
            Y_sc_list.append(Yw_sc[:N_train][:, None, :])      # (nW, 1, H)
            Y_raw_list.append(Yw_raw[:N_train][:, None, :])    # (nW, 1, H) raw counts
            CID_list.append(np.full((min(N_train, Xw_sc.shape[0]),), node, dtype=np.int64))

    if not X_list:
        raise RuntimeError("No training windows after split; check SEQ_LEN/horizon/TRAIN_LAST_DAY.")
    X_tr    = np.concatenate(X_list, axis=0)
    Y_tr_sc = np.concatenate(Y_sc_list, axis=0)
    Y_tr_rw = np.concatenate(Y_raw_list, axis=0)
    CIDs    = np.concatenate(CID_list, axis=0)                 # (num_samples,)

    # surge flags/weights based on raw future maxima
    storm_prior = _load_storm_prior(SURGE_CSV_PATH, train_last_day=train_last_day)
    y_max = np.max(Y_tr_rw.reshape(len(Y_tr_rw), -1), axis=1)
    thr = np.quantile(y_max, max(0.0, 1.0 - storm_prior))
    is_surge = (y_max >= thr).astype(np.float32)
    W_tr = np.where(is_surge > 0.5, W_SURGE, 1.0).astype(np.float32)

    aux = {
        "X_all_scaled": X_all,
        "y_scaled": y_sc,
        "timestamps": ts,
        "days": days,
        "day_end_idx": day_end_idx,
        "storm_prior": float(storm_prior),
        "surge_thr_raw": float(thr),
    }
    return X_tr, Y_tr_sc, Y_tr_rw, W_tr, is_surge, Din, scalers, aux, CIDs

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
            pred_sc = infer(model, Xin_i, scalers, county_id=i)[0, 0]  # (H,)
            pred = np.clip(np.nan_to_num(pred_sc, nan=0.0, posinf=None, neginf=0.0), 0.0, None)
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

def _per_county_rmse_table_and_plots(df_truth_wide: pd.DataFrame,
                                     df_pred_long: pd.DataFrame,
                                     out_dir: str,
                                     epoch: int,
                                     label: str):
    if df_truth_wide is None or df_truth_wide.empty or df_pred_long is None or df_pred_long.empty:
        return

    # --- Align truth (wide) and pred (long→wide)
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
        return
    tw = tw.loc[common_ts]
    pw = pw.loc[common_ts]

    locs = sorted(set(tw.columns).intersection(pw.columns))
    if len(locs) == 0:
        return

    # --- Build long dataframe with truth/pred (ensure 'location' exists)
    truth_long = (
        tw[locs].stack()
        .rename("truth")
        .reset_index()
        .rename(columns={"level_1": "location"})
    )
    pred_long = (
        pw[locs].stack()
        .rename("pred")
        .reset_index()
        .rename(columns={"level_1": "location"})
    )
    # normalize dtypes
    truth_long["location"] = truth_long["location"].astype(str)
    pred_long["location"]  = pred_long["location"].astype(str)

    df_join = pd.merge(truth_long, pred_long, on=["timestamp", "location"], how="inner")

    # per-location RMSE and attach
    def _rmse(group):
        y_true = group["truth"].to_numpy()
        y_pred = group["pred"].to_numpy()
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return np.nan
        return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
    rmse_by_loc = df_join.groupby("location", as_index=False).apply(_rmse).rename(columns={None: "rmse"})
    df_join = df_join.merge(rmse_by_loc, on="location", how="left")

    # --- Save one CSV with truth/pred/RMSE
    csv_path = os.path.join(out_dir, f"val_truth_pred_rmse_epoch{epoch}_{label}.csv")
    df_join = df_join[["location", "timestamp", "truth", "pred", "rmse"]]
    df_join.to_csv(csv_path, index=False)
    print(f"[Val {label}] Saved truth/pred/RMSE table: {csv_path}")

    # --- Per-county PNGs (kept)
    for loc in locs:
        y_true = tw[loc]
        y_pred = pw[loc]
        common = y_true.index.intersection(y_pred.index)
        if len(common) == 0:
            continue
        df_plot = pd.DataFrame({
            "timestamp": common,
            "truth": y_true.loc[common].values,
            "pred": y_pred.loc[common].values
        }).sort_values("timestamp")

        safe_loc = str(loc).replace("/", "-").replace("\\", "-").replace(" ", "_")
        fig_path = os.path.join(out_dir, f"val_curve_epoch{epoch}_{label}_county_{safe_loc}.png")
        plt.figure(figsize=(8, 3))
        plt.plot(df_plot["timestamp"], df_plot["truth"], label="Truth")
        plt.plot(df_plot["timestamp"], df_plot["pred"], label="Prediction")
        plt.title(f"{label} - {loc}")
        plt.xlabel("timestamp"); plt.ylabel("outage")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    print(f"[Val {label}] Saved county-wise figures to: {out_dir}")

    # --- NEW: one multi-panel figure for ALL locations
    n_loc = len(locs)
    n_cols = min(10, max(1, int(np.ceil(np.sqrt(n_loc)))))
    n_rows = int(np.ceil(n_loc / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.6, n_rows * 2.0), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(n_rows, n_cols)

    for idx, loc in enumerate(locs):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        y_true = tw[loc]
        y_pred = pw[loc]
        common = y_true.index.intersection(y_pred.index)
        if len(common) == 0:
            ax.axis("off")
            continue
        ax.plot(common, y_true.loc[common].values, label="Truth")
        ax.plot(common, y_pred.loc[common].values, label="Pred")
        ax.set_title(str(loc), fontsize=9)
        ax.tick_params(labelsize=7)

    for extra in range(n_loc, n_rows * n_cols):
        r, c = divmod(extra, n_cols)
        axes[r, c].axis("off")

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="upper center", ncol=2, fontsize=9)
        plt.subplots_adjust(top=0.92)

    fig.tight_layout()
    fig_path_all = os.path.join(out_dir, f"val_curves_epoch{epoch}_{label}_ALL_LOCATIONS.png")
    fig.savefig(fig_path_all, dpi=150)
    plt.close(fig)
    print(f"[Val {label}] Saved multi-location figure: {fig_path_all}")



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
    # Track best-by-last-day separately (so normal BEST_BY still works if toggle = False)
    best_lastday = {"rmse": float("inf"), "epoch": -1}
    df_truth_val = truth_wide_for_val(ds_train, aux, horizon, train_last_day)

    def _callback(model, epoch):
        df_pred_val = predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day)

        # ---- existing calm/storm/overall blocks (unchanged) ----
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

        # ---- NEW: compute per-day RMSEs for validation window ----
        per_day = _per_day_rmse(ds_train, df_truth_val, df_pred_val, train_last_day)
        if per_day:
            # Save CSV per epoch (now includes column 'zero_rmse')
            df_day = pd.DataFrame(per_day)
            csv_day = os.path.join(OUT_DIR, f"val_rmse_per_day_epoch{epoch}_{label}.csv")
            df_day.to_csv(csv_day, index=False)

            # Print nicely with all-zero baseline
            def _fmt(v):
                return f"{v:.4f}" if np.isfinite(v) else "nan"

            pretty = ", ".join([f"Day {r['day']}: {_fmt(r['rmse'])} (zero={_fmt(r['zero_rmse'])})" for r in per_day])
            print(f"[Val {label}] Epoch {epoch}: Per-day RMSEs -> {pretty}")
            # Last validation day RMSE (model)
            last_day_rmse = per_day[-1]["rmse"] if np.isfinite(per_day[-1]["rmse"]) else float("inf")
        else:
            last_day_rmse = float("inf")

        # ---- existing prints (unchanged) ----
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

        # ---- Selection value: BEST_BY or LAST-VALIDATION-DAY (toggle) ----
        metric_map = {"overall": rmse_overall, "calm": rmse_calm, "storm": rmse_storm}
        sel = metric_map.get(str(BEST_BY).lower(), rmse_overall)
        sel_val = sel if np.isfinite(sel) else rmse_overall

        # Track normal best (as before)
        if np.isfinite(sel_val) and sel_val < best["rmse"]:
            best["rmse"] = sel_val
            best["epoch"] = epoch
            save_path = os.path.join(OUT_DIR, f"best_enc-dec_{label}.pt")
            if isinstance(model, AveragedModel):
                sd_base = model.module.state_dict()
                sd_swa  = model.state_dict()
            else:
                sd_base = model.state_dict()
                sd_swa  = None

            torch.save(
                {
                    "epoch": epoch,
                    "label": label,
                    "rmse_select": sel_val,
                    "rmse_overall": rmse_overall,
                    "rmse_calm": rmse_calm,
                    "rmse_storm": rmse_storm,
                    "rmse_last_val_day": last_day_rmse,
                    "best_by": str(BEST_BY).lower(),
                    "model_state_dict": sd_base,
                    "model_state_dict_swa": sd_swa,
                },
                save_path,
            )
            print(f"[Val {label}] New best ({BEST_BY}) RMSE {sel_val:.4f} at epoch {epoch}. Saved: {save_path}")

            # (existing day-wise save & figures stay unchanged below)
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

            _per_county_rmse_table_and_plots(df_truth_val, df_pred_val, OUT_DIR, epoch, label)

        # ---- NEW: Optional best-by-last-validation-day selection ----
        if SELECT_BEST_BY_LAST_VAL_DAY and np.isfinite(last_day_rmse) and (last_day_rmse < best_lastday["rmse"]):
            best_lastday["rmse"]  = last_day_rmse
            best_lastday["epoch"] = epoch
            save_path = os.path.join(OUT_DIR, f"best_lastday_{label}.pt")
            if isinstance(model, AveragedModel):
                sd_base = model.module.state_dict()
                sd_swa  = model.state_dict()
            else:
                sd_base = model.state_dict()
                sd_swa  = None

            torch.save(
                {
                    "epoch": epoch,
                    "label": label,
                    "rmse_last_val_day": last_day_rmse,
                    "best_by": "last_val_day",
                    "model_state_dict": sd_base,
                    "model_state_dict_swa": sd_swa,
                },
                save_path,
            )
            print(f"[Val {label}] New best-by-last-day RMSE {last_day_rmse:.4f} at epoch {epoch}. Saved: {save_path}")

        # ---- (keep your optional test logging unchanged) ----
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
# objectives
# =======================================================
def _pinball_loss(yq, y_true, q_list, quantile_weights=None):
    qs = torch.tensor(q_list, device=yq.device, dtype=yq.dtype).view(1,1,1,-1)
    diff = y_true.unsqueeze(-1) - yq
    loss = torch.maximum(qs * diff, (qs - 1.0) * diff)
    if quantile_weights is not None:
        qw = torch.tensor(quantile_weights, device=yq.device, dtype=yq.dtype)
        qw = qw / torch.clamp(qw.sum(), min=1e-8)
        loss = loss * qw.view(1,1,1,-1)
    return loss.mean()

def _pinball_subset(yq, y_true, q_list, q_subset):
    qs = np.array(q_list, dtype=float)
    idx = [int(np.argmin(np.abs(qs - q))) for q in q_subset]
    ysel = yq[..., idx]
    return _pinball_loss(ysel, y_true, [q_list[i] for i in idx])

def _huber(x, delta=1.0):
    absx = torch.abs(x)
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device, dtype=x.dtype))
    lin  = absx - quad
    return 0.5 * quad * quad + delta * lin

def _nb_log_prob(y, mu, theta):
    t1 = torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1.0)
    t2 = theta * (torch.log(theta) - torch.log(theta + mu + 1e-12))
    t3 = y * (torch.log(mu + 1e-12) - torch.log(theta + mu + 1e-12))
    return t1 + t2 + t3

def _zinb_nll(y, logit_pi, raw_mu, raw_theta, sample_weight=None):
    pi   = torch.sigmoid(logit_pi)
    mu   = F.softplus(raw_mu) + 1e-6
    th   = F.softplus(raw_theta) + 1e-6

    log_pi   = -F.softplus(-logit_pi)
    log_1mpi = -F.softplus(logit_pi)

    lognb = _nb_log_prob(y, mu, th)
    is_zero = (y <= 0.0 + 1e-7).float()
    lognb0 = th * (torch.log(th) - torch.log(th + mu))
    loglik_zero = torch.logsumexp(torch.stack([log_pi, log_1mpi + lognb0], dim=-1), dim=-1)
    loglik_pos  = (log_1mpi + lognb)

    loglik = is_zero * loglik_zero + (1.0 - is_zero) * loglik_pos
    nll = -loglik
    if sample_weight is not None:
        sw = sample_weight
        while sw.dim() < nll.dim():
            sw = sw.unsqueeze(-1)
        nll = nll * sw
    return nll.mean(), (pi, mu, th)

# =======================================================
# augmentations
# =======================================================
def _maybe_apply_augments(xb, is_surge_batch):
    if not AUG_ENABLE:
        return xb

    B, T, _, Din = xb.shape
    if AUG_SURGE_ONLY:
        eligible = (is_surge_batch > 0.5)
    else:
        eligible = torch.ones(B, device=xb.device, dtype=torch.bool)
    idx = torch.where(eligible)[0]
    if idx.numel() == 0:
        return xb

    x = xb.clone()

    if torch.rand(1, device=xb.device).item() < AUG_PROB:
        choice = torch.randint(0, 4, (1,), device=xb.device).item()

        if choice == 0:
            std = x[:, :, 0, :].std(dim=1, keepdim=True).clamp_min(1e-6)
            noise = torch.randn_like(x[:, :, 0, :]) * (JITTER_STD_FRAC * std)
            x[idx, :, 0, :] = x[idx, :, 0, :] + noise[idx]

        elif choice == 1:
            scale = 1.0 + (2.0 * AMP_MAX_FRAC) * (torch.rand((idx.numel(), 1, 1), device=x.device) - 0.5)
            x[idx, :, 0, :] = x[idx, :, 0, :] * scale

        elif choice == 2:
            s = 1.0 + (2.0 * WARP_MAX_FRAC) * (torch.rand(idx.numel(), device=x.device) - 0.5)
            feat = x[idx, :, 0, :].permute(0, 2, 1)  # (k, Din, T)
            warped = []
            for i in range(feat.size(0)):
                Ti = int(max(2, round(T * s[i].item())))
                zi = F.interpolate(feat[i:i+1], size=Ti, mode="linear", align_corners=False)
                zi = zi[0]  # (Din, Ti)
                if Ti > T:
                    start = (Ti - T) // 2
                    zi = zi[:, start:start+T]
                elif Ti < T:
                    pad = T - Ti
                    last = zi[:, -1:].expand(-1, pad)
                    zi = torch.cat([zi, last], dim=1)
                warped.append(zi.unsqueeze(0))
            warped = torch.cat(warped, dim=0)        # (k, Din, T)
            x[idx, :, 0, :] = warped.permute(0, 2, 1)

        else:
            if idx.numel() >= 2 and torch.rand(1, device=x.device).item() < CUTMIX_PROB:
                perm = idx[torch.randperm(idx.numel(), device=x.device)]
                Lmin = max(2, int(0.2 * T)); Lmax = max(Lmin+1, int(0.5 * T))
                Lseg = torch.randint(Lmin, Lmax, (1,), device=x.device).item()
                s0 = torch.randint(0, T - Lseg + 1, (1,), device=x.device).item()
                s1 = s0 + Lseg
                x[idx, s0:s1, 0, :] = x[perm, s0:s1, 0, :]

    return x

# =======================================================
# training / inference
# =======================================================
def train_model(X, Y_sc, Y_rw, W, is_surge_all, din, horizon, n_nodes, scalers, eval_callback=None, county_ids=None, num_counties=None):
    if len(X) == 0:
        return None

    if county_ids is None:
        raise RuntimeError("county_ids must be provided for county-specific MoE.")

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y_sc, dtype=torch.float32),
                       torch.tensor(Y_rw, dtype=torch.float32),
                       torch.tensor(W, dtype=torch.float32),
                       torch.tensor(is_surge_all, dtype=torch.float32),
                       torch.tensor(county_ids, dtype=torch.long))

    # Reproducible DataLoader
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=False,
                    worker_init_fn=seed_worker,
                    generator=gen)

    model = TCNForecaster(in_channels=din, horizon=horizon, num_nodes=n_nodes,
                          num_counties=int(num_counties),
                          num_experts=NUM_EXPERTS, emb_dim=COUNTY_EMB_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    # SWA
    if USE_SWA:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_start = int(EPOCHS * SWA_START_FRAC)
        swa_model = AveragedModel(model)
        swa_sched = SWALR(opt, swa_lr=LR * 0.5)

    # phase split
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        in_phase1 = (epoch < PHASE1_EPOCHS)
        for xb, y_sc, y_rw, wb, is_surge_b, cid in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)
            y_sc = y_sc.to(DEVICE)
            y_rw = y_rw.to(DEVICE)
            wb = wb.to(DEVICE)
            is_surge_b = is_surge_b.to(DEVICE)
            cid = cid.to(DEVICE)

            if AUG_ENABLE:
                xb = _maybe_apply_augments(xb, is_surge_b)

            opt.zero_grad()
            out = model(xb, cid)
            loss = 0.0

            if in_phase1:
                if USE_ZINB_HEAD and out["zinb"] is not None and PRETRAIN_MODE == "nb_mean":
                    logit_pi, raw_mu, raw_th = out["zinb"]
                    mu = F.softplus(raw_mu) + 1e-6
                    loss_trend = _huber(mu - y_rw, delta=HUBER_DELTA).mean()
                    loss = loss + loss_trend
                    zero_ind = (y_rw <= 0.0 + 1e-7).float()
                    bce = F.binary_cross_entropy_with_logits(logit_pi, zero_ind)
                    loss = loss + LAMBDA_BCE_ZERO * bce
                if KEEP_QUANTILES and out["q"] is not None and PRETRAIN_MODE == "quantile":
                    yq = out["q"]
                    loss = loss + _pinball_subset(yq, y_sc.squeeze(1), QUANTILES, [0.1, 0.5])
            else:
                if USE_ZINB_HEAD and out["zinb"] is not None:
                    logit_pi, raw_mu, raw_th = out["zinb"]
                    loss_zinb, _ = _zinb_nll(y_rw.squeeze(1), logit_pi.squeeze(1), raw_mu.squeeze(1), raw_th.squeeze(1),
                                             sample_weight=wb)
                    loss = loss + LAMBDA_ZINB * loss_zinb

                if KEEP_QUANTILES and out["q"] is not None and len(Q_HIGH) > 0:
                    yq = out["q"]
                    loss_hi = _pinball_subset(yq, y_sc.squeeze(1), QUANTILES, Q_HIGH)
                    loss = loss + LAMBDA_QHI * loss_hi

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        sched.step(epoch_loss)

        if USE_SWA and (epoch >= swa_start):
            swa_model.update_parameters(model)
            swa_sched.step()

        print(f"[Epoch {epoch+1}/{EPOCHS}] Phase={'1' if in_phase1 else '2'} | TrainLoss={epoch_loss:.6f} | LR={opt.param_groups[0]['lr']:.2e}")

        if eval_callback is not None:
            eval_callback(model if not USE_SWA or epoch < swa_start else swa_model, epoch + 1)

    if USE_SWA:
        _update_bn_on_device(dl, swa_model, DEVICE)
        return swa_model
    return model

@torch.no_grad()
def infer(model, Xin, scalers, county_id: int):
    model.eval()
    if isinstance(Xin, np.ndarray):
        B = Xin.shape[0]
    else:
        B = Xin.size(0)
    cid = torch.full((B,), int(county_id), dtype=torch.long, device=DEVICE)
    out = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE), cid)
    if INFER_POINT == "zinb_mean" and out["zinb"] is not None:
        logit_pi, raw_mu, raw_th = out["zinb"]
        pi = torch.sigmoid(logit_pi)
        mu = F.softplus(raw_mu) + 1e-6
        y_point = (1.0 - pi) * mu
        out_np = y_point.cpu().numpy()
        return out_np
    else:
        yq = out["q"]
        if yq is None:
            logit_pi, raw_mu, raw_th = out["zinb"]
            pi = torch.sigmoid(logit_pi)
            mu = F.softplus(raw_mu) + 1e-6
            return ((1.0 - pi) * mu).cpu().numpy()
        qs = np.array(QUANTILES, dtype=float)
        idx = int(np.argmin(np.abs(qs - float(PRED_QUANTILE))))
        y_sc = yq[..., idx]  # (1,N,H)
        y_mu = float(scalers["y_mu"].flatten()[0])
        y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0
        y = (y_sc * y_sd) + y_mu
        return y.cpu().numpy()

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
        pred = infer(model, Xin_i, scalers, county_id=i)[0, 0]
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)
        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))
    return pd.concat(rows, ignore_index=True)

# =======================================================
# main (pipeline preserved: plots, csvs, metrics)
# =======================================================
def main():
    # Fast, reproducible seeding by default; flip STRICT_DETERMINISM for bit-strict runs
    set_seed(SEED, strict=STRICT_DETERMINISM)

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
    X24, Y24_sc, Y24_rw, W24, S24, Din, scalers24, aux24, CIDs24 = prepare_training_split(ds_train, horizon=H24, train_last_day=TRAIN_LAST_DAY)
    X48, Y48_sc, Y48_rw, W48, S48, _,   scalers48, aux48,  CIDs48 = prepare_training_split(ds_train, horizon=H48, train_last_day=TRAIN_LAST_DAY)
    n_nodes = 1
    num_counties = len(ds_train.location.values)

    # ---- 24h
    val_cb24 = make_val_callback_rolling(ds_train, scalers24, H24, aux24, label="24h",
                                         train_last_day=TRAIN_LAST_DAY,
                                         ts_future_test=ts24, df_truth_test=df_t24, zero_rmse_test=zero_rmse_24)
    mdl24 = train_model(X24, Y24_sc, Y24_rw, W24, S24, Din, H24, n_nodes, scalers24,
                        eval_callback=val_cb24, county_ids=CIDs24, num_counties=num_counties)

    # ---- 48h
    val_cb48 = make_val_callback_rolling(ds_train, scalers48, H48, aux48, label="48h",
                                         train_last_day=TRAIN_LAST_DAY,
                                         ts_future_test=ts48, df_truth_test=df_t48, zero_rmse_test=zero_rmse_48)
    mdl48 = train_model(X48, Y48_sc, Y48_rw, W48, S48, Din, H48, n_nodes, scalers48,
                        eval_callback=val_cb48, county_ids=CIDs48, num_counties=num_counties)

    # ----- Final test with best checkpoint if present -----
    def _finalize(model, scalers, H, ts_future, df_truth, tag, zero_rmse):
        # best_path = os.path.join(OUT_DIR, f"best_enc-dec_{tag}.pt")
        best_path = os.path.join(OUT_DIR, f"best_lastday_{tag}.pt")
        if not os.path.exists(best_path):
            best_path = os.path.join(OUT_DIR, f"best_enc-dec_{tag}.pt")

        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=DEVICE)
            sd_base = ckpt.get("model_state_dict", {})
            sd_swa = ckpt.get("model_state_dict_swa", None)

            loaded = False
            if isinstance(model, AveragedModel) and sd_swa is not None:
                loaded = _load_ckpt_flex(model, sd_swa, strict=True)

            if not loaded and sd_base:
                loaded = _load_ckpt_flex(model, sd_base, strict=True)

            if loaded:
                print(f"[{tag}] Loaded best checkpoint (epoch {ckpt.get('epoch')}, "
                      f"val-selected RMSE={ckpt.get('rmse_select', ckpt.get('rmse', float('nan'))):.4f})")
            else:
                print(f"[{tag}] Warning: failed to load best checkpoint with flexible loader.")

        df_pred = predict_per_county(ds_train, model, scalers, H, ts_future)
        rmse = per_county_rmse_mean(df_truth, df_pred)
        print(f"[Test {tag}] RMSE={rmse:.4f} (all-zero RMSE={zero_rmse:.4f})")

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