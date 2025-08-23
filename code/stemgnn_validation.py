# demo_seq2seq_with_stemgnn.py
import os
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import xarray as xr
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from demo_evaluate import load_truth_wide, per_county_rmse_mean

# ---------------- config ----------------
SPLIT_DIR = "data/"
OUT_DIR   = "results_seq/"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN       = 24
BATCH_SIZE    = 16
EPOCHS        = 20
LR            = 1e-4
WEIGHT_DECAY  = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# === NEW: temporal dropout probability (drop whole timesteps during training) ===
TEMPORAL_DROP_PROB = 0.05  # e.g., 5% of timesteps zeroed per window

# --------- StemGNN hyperparams (lightweight, tunable) ----------
STEM_D_MODEL  = 64     # feature hidden width
STEM_K_ORDER  = 2      # graph diffusion orders (0..K)
STEM_NBLOCKS  = 2      # stacked spatio-temporal blocks
STEM_D_EMB    = 32     # node embedding dim (for adaptive adjacency)
STEM_DROPOUT  = 0.3

# === NEW: day split for in-train validation (train on days 1..80, validate on the rest) ===
TRAIN_LAST_DAY = 88

# ---------------- utils (unchanged) ----------------
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
    """
    Build (X,y) windows for one location (single county).
    X_loc: (T, D), y_loc: (T,)
    Returns Xw: (N, seq_len, D), Yw: (N, horizon)
    """
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return np.empty((0, seq_len, X_loc.shape[1]), dtype=float), np.empty((0, horizon), dtype=float)
    Xw, Yw = [], []
    for i in range(N):
        Xw.append(X_loc[i:i+seq_len])                    # past seq_len
        Yw.append(y_loc[i+seq_len:i+seq_len+horizon])    # next horizon
    return np.asarray(Xw, dtype=float), np.asarray(Yw, dtype=float)

# -------------------- StemGNN block --------------------
class _TemporalConv(nn.Module):
    """Depthwise temporal 1D conv over the sequence axis (per node)."""
    def __init__(self, c, dropout=0.0, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(c, c, kernel_size=kernel_size, padding=padding, groups=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        x_bn = x.reshape(B * N, T, C).permute(0, 2, 1)  # (B*N, C, T)
        y = self.conv(x_bn)                             # (B*N, C, T)
        y = self.act(y)
        y = self.dropout(y)
        y = y.permute(0, 2, 1).reshape(B, T, N, C)     # (B, T, N, C)
        return y

class _AdaptiveAdjacency(nn.Module):
    """Learn adaptive adjacency via node embeddings."""
    def __init__(self, num_nodes, d_emb):
        super().__init__()
        self.E = nn.Parameter(torch.randn(num_nodes, d_emb))

    def forward(self):
        # A = softmax(ReLU(E E^T)) row-wise
        E = self.E
        A = torch.relu(E @ E.t())               # (N, N)
        A = torch.softmax(A, dim=1)
        return A

class _GraphDiffusion(nn.Module):
    """Multi-order diffusion on nodes with learnable mixing."""
    def __init__(self, k_order, d_model, dropout=0.0):
        super().__init__()
        self.k_order = k_order
        self.alpha = nn.Parameter(torch.ones(k_order + 1))  # weights for A^0..A^K
        self.dropout = nn.Dropout(dropout)
        # optional mixing after diffusion
        self.mix = nn.Linear(d_model, d_model)

    def forward(self, x, A):
        # x: (B, T, N, C), A: (N, N)
        B, T, N, C = x.shape
        # Precompute powers of A
        A_pows = [torch.eye(N, device=x.device, dtype=x.dtype), A]
        for _ in range(2, self.k_order + 1):
            A_pows.append(A_pows[-1] @ A)
        # Diffusion sum_k alpha_k * (A^k X)
        # Apply per time step with einsum over nodes
        agg = 0.0
        for k, Ak in enumerate(A_pows):
            # y = Ak @ x (over node dim)
            y = torch.einsum('ij,btjc->btic', Ak, x)   # (B, T, N, C)
            agg = agg + self.alpha[k] * y
        out = self.dropout(self.mix(agg))
        return out

class _StemBlock(nn.Module):
    """One spatio-temporal block: TemporalConv -> GraphDiffusion -> Residual + Norm."""
    def __init__(self, d_in, d_model, k_order, num_nodes, d_emb, dropout=0.0):
        super().__init__()
        self.in_proj  = nn.Linear(d_in, d_model)
        self.tconv    = _TemporalConv(d_model, dropout=dropout)
        self.adaptA   = _AdaptiveAdjacency(num_nodes, d_emb)
        self.gdiff    = _GraphDiffusion(k_order, d_model, dropout=dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, N, F_in)
        h0 = self.in_proj(x)                 # (B, T, N, d_model)
        ht = self.tconv(h0)                  # temporal modeling
        A  = self.adaptA()                   # (N, N)
        hg = self.gdiff(ht, A)               # (B, T, N, d_model)
        h  = self.out_norm(h0 + hg)          # residual
        return h

# -------------------- StemGNN Forecaster --------------------
class STGNNForecaster(nn.Module):
    """
    StemGNN-style spatio-temporal forecaster matching your original API.
    Input:  x (B, T, N, F_in)
    Output: (B, N, H) using the last-time-step representation.
    """
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels

        blocks = []
        d_in = in_channels
        for _ in range(STEM_NBLOCKS):
            blocks.append(
                _StemBlock(
                    d_in=d_in,
                    d_model=STEM_D_MODEL,
                    k_order=STEM_K_ORDER,
                    num_nodes=num_nodes,
                    d_emb=STEM_D_EMB,
                    dropout=STEM_DROPOUT,
                )
            )
            d_in = STEM_D_MODEL
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Linear(STEM_D_MODEL, horizon)

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        assert N == self.num_nodes and F == self.in_channels, \
            f"StemGNN input shape mismatch: got N={N},F={F}, expected N={self.num_nodes},F={self.in_channels}"

        h = x
        for blk in self.blocks:
            h = blk(h)              # (B, T, N, d_model)

        # last = h[:, -1, :, :]       # (B, N, d_model)
        # y    = self.readout(last)   # (B, N, H)
        # return y
        pooled = h.mean(dim=1)  # (B, N, d_model)   # simple temporal average
        y = self.readout(pooled)  # (B, N, H)
        return y

# ---------------- training/infer (unchanged API) ----------------
def train_model(X, Y, din, horizon, n_nodes, eval_callback=None):
    """
    X: (num_samples, SEQ_LEN, N, din)
    Y: (num_samples, N, horizon)
    """
    if len(X) == 0:
        return None
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = STGNNForecaster(in_channels=din, horizon=horizon, num_nodes=n_nodes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.SmoothL1Loss(beta=1.0)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)  # xb: (B, T, N, F), yb: (B, N, H)

            # === NEW: Temporal Dropout — randomly zero a few timesteps per window ===
            if TEMPORAL_DROP_PROB > 0.0:
                with torch.no_grad():
                    # Mask shape (B, T, 1, 1): same mask applied across all nodes & features per dropped timestep
                    drop_mask = (torch.rand(xb.size(0), xb.size(1), 1, 1, device=xb.device) < TEMPORAL_DROP_PROB).float()
                    xb = xb * (1.0 - drop_mask)

            opt.zero_grad()
            pred = model(xb)                       # (B, N, H)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s")
        if eval_callback is not None:
            eval_callback(model, epoch + 1)

    return model

@torch.no_grad()
def infer(model, Xin):
    """
    Xin: (1, SEQ_LEN, N, F)
    Returns: (1, N, H)
    """
    model.eval()
    out = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE))
    return out.cpu().numpy()

# ---------------- NEW: split-aware helpers (train on days 1..80; val day-ahead on the rest) ----------------
def _compute_day_indices(ts: np.ndarray):
    """
    Map each timestamp to a day index starting at 1.
    Returns: days (T,), unique_days (list[pd.Timestamp]), day_end_idx (list[int])
    """
    ts_pd = pd.to_datetime(ts)
    days_norm = ts_pd.normalize()
    uniq = days_norm.unique()
    # sort to be sure
    uniq = pd.DatetimeIndex(sorted(uniq))
    # map day number starting at 1
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

def prepare_training_split(ds_train, horizon, train_last_day=80):
    """
    Build training windows **only** whose targets lie fully within days <= train_last_day.
    Returns:
      X_tr: (num_windows, SEQ_LEN, N, Din)
      Y_tr: (num_windows, N, horizon)
      Din, scalers (fit on train split)
      aux: dict with arrays used for validation (scaled series, timestamps, etc.)
    """
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    ts = pd.to_datetime(ds_train.timestamp.values)
    days, uniq_days, day_end_idx = _compute_day_indices(ts)
    T, N, F = w.shape
    Din = 1 + F

    scalers = _fit_scalers_train_only(y, w, days, train_last_day)

    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)       # (T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F)    # (T, N, F)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)                               # (T, N, Din)

    N_total = T - SEQ_LEN - horizon + 1
    if N_total <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    day_last_idx = day_end_idx[train_last_day - 1]
    N_train = max(0, day_last_idx - SEQ_LEN - horizon + 1)

    X_list, Y_list = [], []
    for node in range(N):
        x_loc_full = X_all[:, node, :]                             # (T, Din)
        Xw, Yw = build_windows(x_loc_full, y_sc[:, node], SEQ_LEN, horizon)  # (N_total, SEQ_LEN, Din), (N_total, H)
        if len(Xw):
            X_list.append(Xw[:N_train])
            Y_list.append(Yw[:N_train])

    if not X_list:
        raise RuntimeError("No training windows after split; check SEQ_LEN/horizon/TRAIN_LAST_DAY.")
    X_tr = np.stack(X_list, axis=2)  # (nW, SEQ_LEN, N, Din)
    Y_tr = np.stack(Y_list, axis=2)  # (nW, H, N)
    Y_tr = np.swapaxes(Y_tr, 1, 2)   # -> (nW, N, H)

    aux = {
        "X_all_scaled": X_all,     # (T, N, Din)
        "y_scaled": y_sc,          # (T, N)
        "timestamps": ts,          # (T,)
        "days": days,              # (T,)
        "day_end_idx": day_end_idx # list[int]
    }
    return X_tr, Y_tr, Din, scalers, aux

# ---------------- NEW: day-ahead rolling validation ----------------
@torch.no_grad()
def predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day=80):
    """
    Rolling validation:
      For each day d ∈ [train_last_day, last_day-1], use the **last SEQ_LEN hours of day d**
      to predict the next `horizon` hours starting at the first hour of day d+1.
    Returns a long df with columns [timestamp, location, pred] for ALL validation days.
    """
    model.eval()
    X_all = aux["X_all_scaled"]           # (T, N, Din), scaled with train scalers
    ts    = aux["timestamps"]             # (T,)
    days  = aux["days"]                   # (T,)
    day_end_idx = aux["day_end_idx"]
    N = X_all.shape[1]
    locs = list(map(str, ds_train.location.values))

    rows = []
    last_day = int(days.max())
    for d in range(train_last_day, last_day):
        end_idx = day_end_idx[d - 1]           # last index of day d
        start_ctx = end_idx - (SEQ_LEN - 1)
        fut_start = end_idx + 1
        fut_end   = fut_start + horizon - 1
        if start_ctx < 0 or fut_end >= len(ts):
            continue  # skip if not enough context or future overflow

        Xin = X_all[start_ctx:end_idx+1, :, :]            # (SEQ_LEN, N, Din)
        Xin = Xin.reshape(1, SEQ_LEN, N, X_all.shape[2])  # (1, SEQ_LEN, N, Din)

        pred_sc = infer(model, Xin)[0]                    # (N, H) scaled
        # invert scaling to original
        y_mu = float(scalers["y_mu"].flatten()[0])
        y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0
        pred = (pred_sc * y_sd) + y_mu
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

        ts_future = pd.to_datetime(ts[fut_start:fut_start+horizon])
        for i, county in enumerate(locs):
            rows.append(pd.DataFrame({
                "timestamp": ts_future,
                "location": county,
                "pred": pred[i].astype(float)
            }))

    if rows:
        df_pred = pd.concat(rows, ignore_index=True)
    else:
        df_pred = pd.DataFrame(columns=["timestamp", "location", "pred"])
    return df_pred

def truth_wide_for_val(ds_train, aux, horizon, train_last_day=80):
    """
    Build the truth wide dataframe restricted to the union of all validation future timestamps.
    """
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

    y = ds_train.out.transpose("timestamp", "location").values.astype(float)  # (T, N)
    locs = list(map(str, ds_train.location.values))
    df = pd.DataFrame(y[fut_indices, :], columns=locs, index=ts_fut).reset_index().rename(columns={"index": "timestamp"})
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

# -------- NEW: helper for all-zero baseline on a given truth frame --------
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

# -------- NEW: unified callback (rolling val + test; save best by val RMSE) --------
def make_val_callback_rolling(ds_train, scalers, horizon, aux, label,
                              train_last_day=80,
                              ts_future_test=None, df_truth_test=None, zero_rmse_test=float("nan")):
    """
    Evaluate each epoch on rolling day-ahead validation (days 81..end) **and** on the test set.
    Prints:
      - Rolling RMSE=xxx.xxxx (all-zero RMSE=yyy.yyyy)
      - Testing RMSE=aaa.aaaa (all-zero RMSE=bbb.bbbb)
    Saves best checkpoint (by validation RMSE).
    """
    best = {"rmse": float("inf"), "epoch": -1}
    df_truth_val = truth_wide_for_val(ds_train, aux, horizon, train_last_day)

    # Pre-build an all-zero baseline for validation timestamps/locations
    if not df_truth_val.empty:
        zero_rmse_val = _all_zero_rmse_for_truth(df_truth_val)
    else:
        zero_rmse_val = float("nan")

    def _callback(model, epoch):
        # ---- Validation (rolling) ----
        df_pred_val = predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day)
        if df_pred_val.empty or df_truth_val.empty:
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE=nan (all-zero RMSE=nan)")
        else:
            rmse = per_county_rmse_mean(df_truth_val, df_pred_val)
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE={rmse:.4f} (all-zero RMSE={zero_rmse_val:.4f})")

            # Save best by validation RMSE
            if np.isfinite(rmse) and rmse < best["rmse"]:
                best["rmse"] = rmse
                best["epoch"] = epoch
                save_path = os.path.join(OUT_DIR, f"best_stgnn_{label}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "label": label,
                        "rmse": rmse,
                        "model_state_dict": model.state_dict(),
                    },
                    save_path,
                )
                print(f"[Val {label}] New best RMSE {rmse:.4f} at epoch {epoch}. Saved: {save_path}")

        # ---- Test set evaluation at each validation step ----
        if ts_future_test is not None and df_truth_test is not None and len(ts_future_test) > 0:
            df_pred_test = predict_per_county(ds_train, model, scalers, horizon, ts_future_test)
            rmse_test = per_county_rmse_mean(df_truth_test, df_pred_test)
            print(f"[Val {label}] Epoch {epoch}: Testing RMSE={rmse_test:.4f} (all-zero RMSE={zero_rmse_test:.4f})")

    return _callback

# ---------------- pipeline helpers (replaced with split-aware builder) ----------------
def prepare_training(ds_train, horizon):
    """
    (Kept for compatibility but now calls the split-aware builder.)
    """
    X, Y, Din, scalers, aux = prepare_training_split(ds_train, horizon, TRAIN_LAST_DAY)
    return X, Y, Din, scalers, aux

# ---------------- test-time prediction (unchanged) ----------------
def predict_per_county(ds_train, model, scalers, horizon, ts_future):
    """
    Predict next 'horizon' for ALL counties at once using last SEQ_LEN hours of train.
    """
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    T, N, F = w.shape
    Din = 1 + F

    # Scale inputs
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)    # (T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F) # (T, N, F)

    # Build last SEQ_LEN window for all nodes: (1, SEQ_LEN, N, Din)
    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)   # (T, N, Din)
    Xin = X_all[-SEQ_LEN:]                                    # (SEQ_LEN, N, Din)
    Xin = Xin.reshape(1, SEQ_LEN, N, Din)                     # (1, SEQ_LEN, N, Din)

    # Predict (scaled)
    pred_sc = infer(model, Xin)[0]                            # (N, H)

    # Invert scaling
    pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
    pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

    # Assemble DataFrame
    rows = []
    for i, county in enumerate(locs):
        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred[i].astype(float)}))
    return pd.concat(rows, ignore_index=True)

# ---------------- main ----------------
def main():
    # Load train + test
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))
    # ds_train = open_ds(os.path.join(SPLIT_DIR, "train_50days.nc"))
    # ds_test24 = xr.open_dataset(os.path.join(SPLIT_DIR, "valid_1day.nc"))
    # ds_test48 = xr.open_dataset(os.path.join(SPLIT_DIR, "valid_2days.nc"))

    # Ensure features exist (and implicitly use all of them)
    _ = get_all_features(ds_train)

    # Horizons and timestamps
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    # Truth data for evaluation (test)
    df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))
    # df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "valid_1day.nc"))
    # df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "valid_2days.nc"))

    # Pre-compute all-zero RMSE for test sets (NEW)
    zero_rmse_24 = _all_zero_rmse_for_truth(df_t24)
    zero_rmse_48 = _all_zero_rmse_for_truth(df_t48)

    # ==== 24h ====
    X24, Y24, Din, scalers24, aux24 = prepare_training(ds_train, horizon=H24)
    n_nodes = ds_train.sizes["location"]
    val_cb24 = make_val_callback_rolling(
        ds_train, scalers24, H24, aux24, label="24h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts24, df_truth_test=df_t24, zero_rmse_test=zero_rmse_24
    )
    mdl24 = train_model(X24, Y24, Din, H24, n_nodes, eval_callback=val_cb24)

    # ==== 48h ====
    X48, Y48, _, scalers48, aux48 = prepare_training(ds_train, horizon=H48)
    val_cb48 = make_val_callback_rolling(
        ds_train, scalers48, H48, aux48, label="48h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts48, df_truth_test=df_t48, zero_rmse_test=zero_rmse_48
    )
    mdl48 = train_model(X48, Y48, Din, H48, n_nodes, eval_callback=val_cb48)

    # ----- Test evaluation using best models, then write CSVs (print with all-zero baseline) -----
    # 24h
    if mdl24 is not None:
        best24_path = os.path.join(OUT_DIR, "best_stgnn_24h.pt")
        if os.path.exists(best24_path):
            ckpt24 = torch.load(best24_path, map_location=DEVICE)
            try:
                mdl24.load_state_dict(ckpt24["model_state_dict"])
                rmse24_val = ckpt24.get("rmse", None)
                ep24 = ckpt24.get("epoch", None)
                if isinstance(rmse24_val, (float, int)):
                    print(f"[24h] Loaded best checkpoint (epoch {ep24}, Val RMSE={rmse24_val:.4f})")
                else:
                    print(f"[24h] Loaded best checkpoint (epoch {ep24})")
            except Exception as e:
                print(f"[24h] Warning: failed to load best checkpoint: {e}")
        # Predict on test set
        df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24)
        test_rmse_24 = per_county_rmse_mean(df_t24, df24)
        print(f"[Test 24h] Testing RMSE={test_rmse_24:.4f} (all-zero RMSE={zero_rmse_24:.4f})")
    else:
        df24 = None

    # 48h
    if mdl48 is not None:
        best48_path = os.path.join(OUT_DIR, "best_stgnn_48h.pt")
        if os.path.exists(best48_path):
            ckpt48 = torch.load(best48_path, map_location=DEVICE)
            try:
                mdl48.load_state_dict(ckpt48["model_state_dict"])
                rmse48_val = ckpt48.get("rmse", None)
                ep48 = ckpt48.get("epoch", None)
                if isinstance(rmse48_val, (float, int)):
                    print(f"[48h] Loaded best checkpoint (epoch {ep48}, Val RMSE={rmse48_val:.4f})")
                else:
                    print(f"[48h] Loaded best checkpoint (epoch {ep48})")
            except Exception as e:
                print(f"[48h] Warning: failed to load best checkpoint: {e}")
        # Predict on test set
        df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48)
        test_rmse_48 = per_county_rmse_mean(df_t48, df48)
        print(f"[Test 48h] Testing RMSE={test_rmse_48:.4f} (all-zero RMSE={zero_rmse_48:.4f})")
    else:
        df48 = None

    # Fallbacks (shouldn't trigger)
    if df24 is None:
        locs = list(map(str, ds_train.location.values))
        df24 = pd.DataFrame({"timestamp": np.repeat(ts24, len(locs)),
                             "location": np.tile(locs, len(ts24)),
                             "pred": 0.0})
    if df48 is None:
        locs = list(map(str, ds_train.location.values))
        df48 = pd.DataFrame({"timestamp": np.repeat(ts48, len(locs)),
                             "location": np.tile(locs, len(ts48)),
                             "pred": 0.0})

    # Save
    out24 = os.path.join(OUT_DIR, "stemgnn_pred_24h.csv")  # kept filename to avoid changing other code
    out48 = os.path.join(OUT_DIR, "stemgnn_pred_48h.csv")
    df24.to_csv(out24, index=False)
    df48.to_csv(out48, index=False)

    print("Saved predictions:")
    print(f"  {out24}")
    print(f"  {out48}")

    # Clean up
    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
