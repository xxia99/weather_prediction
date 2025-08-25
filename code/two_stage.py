# stemgnn_two_stage_zero_inflated.py
import os
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from demo_evaluate import load_truth_wide, per_county_rmse_mean

# ---------------- config ----------------
SPLIT_DIR = "data/"
OUT_DIR   = "results_two_stage/"
os.makedirs(OUT_DIR, exist_ok=True)

# core training hyperparameters
SEQ_LEN       = 24
BATCH_SIZE    = 64
EPOCHS        = 40
LR            = 5e-4
WEIGHT_DECAY  = 5e-5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# === temporal dropout probability (drop whole timesteps during training) ===
TEMPORAL_DROP_PROB = 0.0

# === PatchTST encoder hyperparams ===
PATCH_LEN      = 2
PATCH_STRIDE   = None
PATCH_D_MODEL  = 64
PATCH_NHEAD    = 4
PATCH_NLAYERS  = 2
PATCH_DROPOUT  = 0.05
PATCH_FF       = 4 * PATCH_D_MODEL

# --------- StemGNN hyperparams ----------
STEM_D_MODEL  = 64
STEM_K_ORDER  = 2
STEM_NBLOCKS  = 2
STEM_D_EMB    = 32
STEM_DROPOUT  = 0.03

# === Day split for in-train validation ===
TRAIN_LAST_DAY = 80

# === Event-stratified validation settings ===
EVENT_METHOD   = "mean"
STORM_TOP_Q    = 0.80
CALM_BOTTOM_Q  = 0.50
CALM_N         = 10
BEST_BY        = "overall"  # "overall" | "calm" | "storm"

# === Zero-inflated two-stage settings ===
ZERO_LABEL_THR_ORIG = 0.0         # classify y>0 as non-zero
CLS_PROB_THR        = 0.5         # threshold on classifier probability at inference
W_CLS               = 1.0         # classification loss weight
W_REG               = 1.0         # regression loss weight (only where y>0)
CLS_POS_WEIGHT_MODE = "auto"      # "auto" or float (e.g., 5.0)

# ---------------- utils ----------------
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

# -------------------- PatchTST encoder --------------------
class PatchTSTEncoder(nn.Module):
    """
    Channel-last patching along time. For each node independently:
      Input  : (B, T, N, F)
      Patch  : split time into tokens of length P (stride S)
      Encode : Transformer over tokens per node
      Unpatch: repeat each token embedding over its P time steps -> (B, T_pad, N, D)
      Trim   : keep first T steps -> (B, T, N, D)
    """
    def __init__(self, in_channels, patch_len=PATCH_LEN, stride=PATCH_STRIDE,
                 d_model=PATCH_D_MODEL, nhead=PATCH_NHEAD, nlayers=PATCH_NLAYERS,
                 dropout=PATCH_DROPOUT, ff=PATCH_FF):
        super().__init__()
        self.in_channels = in_channels
        self.patch_len   = patch_len
        self.stride      = patch_len if stride is None else stride
        self.d_model     = d_model

        self.proj = nn.Linear(in_channels * self.patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        P, S = self.patch_len, self.stride
        assert F == self.in_channels, f"PatchTST got F={F}, expected {self.in_channels}"

        # reshape for per-node processing
        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)

        # pad time so we can take integer number of patches
        L = (T + S - 1) // S  # number of tokens if we stride by S
        T_pad = (L - 1) * S + P
        if T_pad < T:
            L += 1
            T_pad = (L - 1) * S + P
        if T_pad > T:
            pad = torch.zeros(B * N, T_pad - T, F, device=x.device, dtype=x.dtype)
            x_bn = torch.cat([x_bn, pad], dim=1)  # (B*N, T_pad, F)

        # build patches with unfold (time axis)
        x_unf = x_bn.unfold(dimension=1, size=P, step=S)          # (B*N, L, P, F)
        x_unf = x_unf.contiguous().view(B * N, L, P * F)          # (B*N, L, P*F)

        # token embedding + transformer
        tokens = self.proj(x_unf)                                 # (B*N, L, d_model)
        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)                              # (B*N, L, d_model)

        # unpatch: repeat each token across P steps
        tokens_rep = tokens.repeat_interleave(P, dim=1)            # (B*N, L*P, d_model)
        # trim to original T (assuming S==P; overlap would need weighted merge)
        steps = tokens_rep[:, :T, :]                               # (B*N, T, d_model)

        # reshape back to (B, T, N, d_model)
        h = steps.view(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B, T, N, d_model)
        return h

# -------------------- StemGNN blocks --------------------
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
        self.mix = nn.Linear(d_model, d_model)

    def forward(self, x, A):
        # x: (B, T, N, C), A: (N, N)
        if self.k_order <= 0:
            return self.dropout(self.mix(x))
        B, T, N, C = x.shape
        I = torch.eye(N, device=x.device, dtype=x.dtype)
        A_pows = [I]
        Ak = A
        for _ in range(1, self.k_order + 1):
            A_pows.append(Ak)
            Ak = Ak @ A
        agg = 0.0
        for k, Ak in enumerate(A_pows):
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
        h0 = self.in_proj(x)                 # (B, T, N, d_model)
        ht = self.tconv(h0)                  # temporal modeling
        A  = self.adaptA()                   # (N, N)
        hg = self.gdiff(ht, A)               # (B, T, N, d_model)
        h  = self.out_norm(h0 + hg)          # residual
        return h

# -------------- Two-stage zero-inflated forecaster --------------
class ZeroInflatedSTGNN(nn.Module):
    """
    Shared encoder (PatchTST -> Stem blocks), two heads:
      - cls_head: outage occurrence logits (B, N, H)
      - reg_head: outage magnitude (scaled space) (B, N, H)
    Inference: gate magnitude by (sigmoid(logits) >= CLS_PROB_THR)
    """
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels

        # Patch encoder
        self.patch = PatchTSTEncoder(
            in_channels=in_channels,
            patch_len=PATCH_LEN,
            stride=PATCH_STRIDE,
            d_model=PATCH_D_MODEL,
            nhead=PATCH_NHEAD,
            nlayers=PATCH_NLAYERS,
            dropout=PATCH_DROPOUT,
            ff=PATCH_FF
        )

        # Stem blocks
        enc_out_dim = getattr(self.patch, "d_model", PATCH_D_MODEL)
        blocks = []
        d_in = enc_out_dim
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

        # Heads
        self.cls_head = nn.Linear(STEM_D_MODEL, horizon)  # logits
        self.reg_head = nn.Linear(STEM_D_MODEL, horizon)  # scaled magnitude (can be negative in z-space)

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        assert N == self.num_nodes and F == self.in_channels, \
            f"Input shape mismatch: got N={N},F={F}, expected N={self.num_nodes},F={self.in_channels}"

        h = self.patch(x)                # (B, T, N, Dp)
        for blk in self.blocks:
            h = blk(h)                   # (B, T, N, Dm)
        last = h[:, -1]                  # (B, N, Dm)

        cls_logits = self.cls_head(last) # (B, N, H)
        reg_out    = self.reg_head(last) # (B, N, H) (scaled)
        return cls_logits, reg_out

# ---------------- helpers for day split & windows ----------------
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

def prepare_training_split(ds_train, horizon, train_last_day=80):
    """
    Build training windows **only** whose targets lie fully within days <= train_last_day.
    Returns:
      X_tr: (num_windows_total, SEQ_LEN, 1, Din)    # one county per sample (N=1)
      Y_tr: (num_windows_total, 1, horizon)         # scaled
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

    # --- per-county windows, concatenated on batch axis; set N=1 ---
    X_list, Y_list = [], []
    for node in range(N):
        x_loc_full = X_all[:, node, :]                             # (T, Din)
        Xw, Yw = build_windows(x_loc_full, y_sc[:, node], SEQ_LEN, horizon)  # (N_total, SEQ_LEN, Din), (N_total, H)
        if len(Xw):
            X_list.append(Xw[:N_train][:, :, None, :])  # -> (nW, SEQ_LEN, 1, Din)
            Y_list.append(Yw[:N_train][:, None, :])     # -> (nW, 1, H)

    if not X_list:
        raise RuntimeError("No training windows after split; check SEQ_LEN/horizon/TRAIN_LAST_DAY.")
    X_tr = np.concatenate(X_list, axis=0)  # (nW*N, SEQ_LEN, 1, Din)
    Y_tr = np.concatenate(Y_list, axis=0)  # (nW*N, 1, H)

    aux = {
        "X_all_scaled": X_all,     # (T, N, Din)
        "y_scaled": y_sc,          # (T, N)
        "timestamps": ts,          # (T,)
        "days": days,              # (T,)
        "day_end_idx": day_end_idx # list[int]
    }
    return X_tr, Y_tr, Din, scalers, aux

# ---------------- validation roll ----------------
@torch.no_grad()
def infer_two_stage(model, Xin):
    """
    Xin: (1, SEQ_LEN, 1, Din)  scaled
    Returns: (prob: (H,), reg_sc: (H,)) both numpy
    """
    model.eval()
    Xin_t = torch.tensor(Xin, dtype=torch.float32, device=DEVICE)
    cls_logits, reg_sc = model(Xin_t)       # (1,1,H), (1,1,H)
    prob = torch.sigmoid(cls_logits)[0, 0].detach().cpu().numpy()
    reg  = reg_sc[0, 0].detach().cpu().numpy()
    return prob, reg

@torch.no_grad()
def predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day=80, prob_thr=CLS_PROB_THR):
    """
    Rolling validation with two-stage gating:
      For each day d ∈ [train_last_day, last_day-1], for each county, use last SEQ_LEN of day d
      to predict next horizon of day d+1. Final pred = (prob>=thr) * reg, then inverse-scale.
    Returns long df: [timestamp, location, pred]
    """
    model.eval()
    X_all = aux["X_all_scaled"]           # (T, N, Din)
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
            Xin_i = X_all[start_ctx:end_idx+1, i, :].reshape(1, SEQ_LEN, 1, X_all.shape[2])  # (1, T, 1, Din)
            prob, reg_sc = infer_two_stage(model, Xin_i)                                     # (H,), (H,)
            mask = (prob >= prob_thr).astype(np.float32)
            pred_sc = reg_sc * mask

            y_mu = float(scalers["y_mu"].flatten()[0])
            y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0
            pred = (pred_sc * y_sd) + y_mu
            pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

            rows.append(pd.DataFrame({
                "timestamp": ts_future,
                "location": county,
                "pred": pred.astype(float)
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

# -------- helper for all-zero baseline on a given truth frame --------
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

# ---- helpers for event selection (calm/storm) ----
def _per_day_outage_score(ds_train, method: str = "mean"):
    ts = ds_train.timestamp.values
    days, _, _ = _compute_day_indices(ts)
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)  # (T, N)
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
    # enforce dtype to avoid merge issues in per_county_rmse_mean
    if not df_truth_sel.empty:
        df_truth_sel["timestamp"] = pd.to_datetime(df_truth_sel["timestamp"])
    if not df_pred_sel.empty:
        df_pred_sel["timestamp"] = pd.to_datetime(df_pred_sel["timestamp"])
    return df_truth_sel, df_pred_sel

def _event_rmse_blocks(ds_train, df_truth_val, df_pred_val,
                       train_last_day, method, storm_top_q, calm_bottom_q, calm_n, seed):
    """
    Returns:
      (rmse_overall, zero_overall, rmse_calm, zero_calm, n_calm_days,
       rmse_storm, zero_storm, n_storm_days)
    """
    # overall
    rmse_overall = per_county_rmse_mean(df_truth_val, df_pred_val) if (not df_truth_val.empty and not df_pred_val.empty) else float("nan")
    zero_overall = _all_zero_rmse_for_truth(df_truth_val) if not df_truth_val.empty else float("nan")

    # calm/storm selection
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

    # calm block
    if len(ts_calm) > 0:
        truth_calm, pred_calm = _filter_truth_pred_by_days(df_truth_val, df_pred_val, ts_calm)
        rmse_calm = per_county_rmse_mean(truth_calm, pred_calm)
        zero_calm = _all_zero_rmse_for_truth(truth_calm)
        n_calm = len(calm_days)
    else:
        rmse_calm, zero_calm, n_calm = float("nan"), float("nan"), 0

    # storm block
    if len(ts_storm) > 0:
        truth_storm, pred_storm = _filter_truth_pred_by_days(df_truth_val, df_pred_val, ts_storm)
        rmse_storm = per_county_rmse_mean(truth_storm, pred_storm)
        zero_storm = _all_zero_rmse_for_truth(truth_storm)
        n_storm = len(storm_days)
    else:
        rmse_storm, zero_storm, n_storm = float("nan"), float("nan"), 0

    return rmse_overall, zero_overall, rmse_calm, zero_calm, n_calm, rmse_storm, zero_storm, n_storm

# --------- Zero/Nonzero ratios ----------
def _pivot_pred_long_to_wide(df_pred_long: pd.DataFrame, locs: list[str]) -> pd.DataFrame:
    if df_pred_long is None or df_pred_long.empty:
        return pd.DataFrame(columns=["timestamp", *locs])
    pivot = df_pred_long.pivot_table(index="timestamp", columns="location", values="pred",
                                     aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(columns=locs, fill_value=0.0).sort_index().reset_index()
    pivot.columns.name = None
    return pivot

def zero_nonzero_ratios(df_truth_wide: pd.DataFrame, df_pred_long: pd.DataFrame):
    """
    Returns:
      zero_to_nonzero, nonzero_to_zero
    """
    if df_truth_wide.empty or df_pred_long.empty:
        return float("nan"), float("nan")
    # Coerce dtype
    df_truth = df_truth_wide.copy()
    df_truth["timestamp"] = pd.to_datetime(df_truth["timestamp"])
    locs = [c for c in df_truth.columns if c != "timestamp"]
    df_pred_wide = _pivot_pred_long_to_wide(df_pred_long, locs)
    df_pred_wide["timestamp"] = pd.to_datetime(df_pred_wide["timestamp"])
    df = df_truth.merge(df_pred_wide, on="timestamp", how="inner", suffixes=("_true", "_pred"))
    if df.empty:
        return float("nan"), float("nan")

    truth = df[[f"{c}_true" for c in locs]].values
    pred  = df[[f"{c}_pred" for c in locs]].values
    truth_nonzero = (truth > 0).astype(np.float32)
    truth_zero    = 1.0 - truth_nonzero
    pred_nonzero  = (pred  > 0).astype(np.float32)
    pred_zero     = 1.0 - pred_nonzero

    n_truth_zero = truth_zero.sum()
    n_truth_nonzero = truth_nonzero.sum()
    if n_truth_zero <= 0:
        z2nz = float("nan")
    else:
        z2nz = float((truth_zero * pred_nonzero).sum() / n_truth_zero)
    if n_truth_nonzero <= 0:
        nz2z = float("nan")
    else:
        nz2z = float((truth_nonzero * pred_zero).sum() / n_truth_nonzero)
    return z2nz, nz2z

def make_val_callback_rolling(ds_train, scalers, horizon, aux, label,
                              train_last_day=80,
                              ts_future_test=None, df_truth_test=None, zero_rmse_test=float("nan")):
    """
    Evaluate each epoch on rolling day-ahead validation (days 81..end) **and** on the test set.
    Prints:
      - Rolling RMSE and all-zero baseline
      - Zero→Nonzero and Nonzero→Zero ratios (with all-zero baselines)
      - Calm-heavy & Storm-heavy RMSEs + all-zero baselines
      - Testing RMSE (with all-zero baseline)
    Saves best checkpoint by metric selected via BEST_BY.
    """
    best = {"rmse": float("inf"), "epoch": -1}
    df_truth_val = truth_wide_for_val(ds_train, aux, horizon, train_last_day)
    zero_rmse_val = _all_zero_rmse_for_truth(df_truth_val) if not df_truth_val.empty else float("nan")

    # All-zero baseline ratios
    locs = [c for c in df_truth_val.columns if c != "timestamp"]
    df_zero_pred = pd.DataFrame({
        "timestamp": np.repeat(df_truth_val["timestamp"].values, len(locs)),
        "location": np.tile(locs, len(df_truth_val)),
        "pred": 0.0
    })
    z2nz_zero, nz2z_zero = zero_nonzero_ratios(df_truth_val, df_zero_pred)

    def _callback(model, epoch):
        df_pred_val = predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day, prob_thr=CLS_PROB_THR)

        # overall metrics
        rmse_overall = per_county_rmse_mean(df_truth_val, df_pred_val) if (not df_truth_val.empty and not df_pred_val.empty) else float("nan")
        z2nz, nz2z = zero_nonzero_ratios(df_truth_val, df_pred_val)

        # event-stratified metrics (with all-zero baselines)
        (rmse_overall_ev, zero_overall,
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

        # Print nicely
        if np.isfinite(rmse_overall):
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE={rmse_overall:.4f} (all-zero={zero_rmse_val:.4f})")
        else:
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE=nan (all-zero=nan)")

        if np.isfinite(z2nz):
            print(f"[Val {label}] Zero→Nonzero={z2nz:.4f} (all-zero={z2nz_zero:.4f}) | Nonzero→Zero={nz2z:.4f} (all-zero={nz2z_zero:.4f})")
        else:
            print(f"[Val {label}] Zero→Nonzero=nan (all-zero={z2nz_zero:.4f}) | Nonzero→Zero=nan (all-zero={nz2z_zero:.4f})")

        if np.isfinite(rmse_calm):
            print(f"[EventEval {label}] Calm-heavy ({n_calm} days): RMSE={rmse_calm:.4f} (all-zero={zero_calm:.4f})")
        else:
            print(f"[EventEval {label}] Calm-heavy: RMSE=nan (all-zero=nan)")

        if np.isfinite(rmse_storm):
            print(f"[EventEval {label}] Storm-heavy ({n_storm} days): RMSE={rmse_storm:.4f} (all-zero={zero_storm:.4f})")
        else:
            print(f"[EventEval {label}] Storm-heavy: RMSE=nan (all-zero=nan)")

        # Which metric selects "best"?
        metric_map = {
            "overall": rmse_overall_ev,  # use the same rmse as event block (equals rmse_overall)
            "calm": rmse_calm,
            "storm": rmse_storm
        }
        sel = metric_map.get(str(BEST_BY).lower(), rmse_overall_ev)
        sel_val = sel if np.isfinite(sel) else rmse_overall_ev

        if np.isfinite(sel_val) and sel_val < best["rmse"]:
            best["rmse"] = sel_val
            best["epoch"] = epoch
            save_path = os.path.join(OUT_DIR, f"best_two_stage_{label}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "label": label,
                    "rmse_select": sel_val,
                    "rmse_overall": rmse_overall_ev,
                    "rmse_calm": rmse_calm,
                    "rmse_storm": rmse_storm,
                    "best_by": str(BEST_BY).lower(),
                    "model_state_dict": model.state_dict(),
                },
                save_path,
            )
            print(f"[Val {label}] New best ({BEST_BY}) RMSE {sel_val:.4f} at epoch {epoch}. Saved: {save_path}")

        # ---- Test set evaluation at each validation step ----
        if ts_future_test is not None and df_truth_test is not None and len(ts_future_test) > 0:
            df_pred_test = predict_per_county(ds_train, model, scalers, horizon, ts_future_test)
            rmse_test = per_county_rmse_mean(df_truth_test, df_pred_test)
            # Baseline ratios for test (all-zero)
            locs_t = [c for c in df_truth_test.columns if c != "timestamp"]
            df_zero_pred_t = pd.DataFrame({
                "timestamp": np.repeat(df_truth_test["timestamp"].values, len(locs_t)),
                "location": np.tile(locs_t, len(df_truth_test)),
                "pred": 0.0
            })
            z2nz_t, nz2z_t = zero_nonzero_ratios(df_truth_test, df_pred_test)
            z2nz_zero_t, nz2z_zero_t = zero_nonzero_ratios(df_truth_test, df_zero_pred_t)
            print(f"[Val {label}] Epoch {epoch}: Testing RMSE={rmse_test:.4f} (all-zero RMSE={_all_zero_rmse_for_truth(df_truth_test):.4f})")
            if np.isfinite(z2nz_t):
                print(f"[Test {label}] Zero→Nonzero={z2nz_t:.4f} (all-zero={z2nz_zero_t:.4f}) | Nonzero→Zero={nz2z_t:.4f} (all-zero={nz2z_zero_t:.4f})")

    return _callback

# ---------------- training/infer ----------------
def train_model(X, Y, din, horizon, n_nodes, scalers, eval_callback=None):
    """
    X: (num_samples, SEQ_LEN, N=1, din)  scaled
    Y: (num_samples, N=1, horizon)       scaled
    """
    if len(X) == 0:
        return None

    # Compute scaled threshold for labels (global scalers)
    y_mu = float(scalers["y_mu"].flatten()[0])
    y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0
    thr_scaled = (ZERO_LABEL_THR_ORIG - y_mu) / y_sd

    # Classification pos_weight (imbalance handling)
    if CLS_POS_WEIGHT_MODE == "auto":
        labels_all = (Y > thr_scaled).astype(np.float32)
        pos = labels_all.sum()
        neg = labels_all.size - pos
        pos_weight_val = float(neg / max(pos, 1.0)) if pos > 0 else 1.0
    else:
        pos_weight_val = float(CLS_POS_WEIGHT_MODE)

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ZeroInflatedSTGNN(in_channels=din, horizon=horizon, num_nodes=n_nodes).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=DEVICE))

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_mse_nonzero = 0.0
        epoch_bce = 0.0
        count_samples = 0
        start_time = time.time()

        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)  # xb: (B, T, 1, F), yb: (B, 1, H)

            if TEMPORAL_DROP_PROB > 0.0:
                with torch.no_grad():
                    drop_mask = (torch.rand(xb.size(0), xb.size(1), 1, 1, device=xb.device) < TEMPORAL_DROP_PROB).float()
                    xb = xb * (1.0 - drop_mask)

            # Labels in scaled space
            with torch.no_grad():
                labels = (yb > thr_scaled).float()               # (B,1,H)

            opt.zero_grad()
            cls_logits, reg_pred = model(xb)                     # (B,1,H), (B,1,H)

            # Classification loss
            loss_cls = bce(cls_logits, labels)

            # Regression loss only over nonzero targets
            mask = labels
            denom = mask.sum().clamp_min(1.0)
            mse_nonzero = ((reg_pred - yb)**2 * mask).sum() / denom

            loss = W_CLS * loss_cls + W_REG * mse_nonzero
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = xb.size(0)
            epoch_loss += loss.item() * bs
            epoch_mse_nonzero += mse_nonzero.item() * bs
            epoch_bce += loss_cls.item() * bs
            count_samples += bs

        epoch_loss /= max(1, count_samples)
        epoch_mse_nonzero /= max(1, count_samples)
        epoch_bce /= max(1, count_samples)
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | MSE_nonzero: {epoch_mse_nonzero:.6f} | BCE: {epoch_bce:.6f} | Time: {elapsed:.2f}s")

        if eval_callback is not None:
            eval_callback(model, epoch + 1)

    return model

# ---------------- test-time prediction (two-stage gating, per-county) ----------------
@torch.no_grad()
def predict_per_county(ds_train, model, scalers, horizon, ts_future, prob_thr=CLS_PROB_THR):
    """
    Predict next 'horizon' for ALL counties, running the model per-county (N=1) with gating.
    """
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    T, N, F = w.shape
    Din = 1 + F

    # Scale inputs with train scalers
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)    # (T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F) # (T, N, F)

    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)   # (T, N, Din)

    rows = []
    for i, county in enumerate(locs):
        if T < SEQ_LEN:
            pred_sc = np.zeros(horizon, dtype=float)
        else:
            Xin_i = X_all[-SEQ_LEN:, i, :].reshape(1, SEQ_LEN, 1, Din)  # (1, SEQ_LEN, 1, Din)
            prob, reg_sc = infer_two_stage(model, Xin_i)
            mask = (prob >= prob_thr).astype(np.float32)
            pred_sc = reg_sc * mask

        pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))
    return pd.concat(rows, ignore_index=True)

# ---------------- main ----------------
def main():
    # Load train + test
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

    # ==== 24h ====
    X24, Y24, Din, scalers24, aux24 = prepare_training_split(ds_train, horizon=H24, train_last_day=TRAIN_LAST_DAY)
    n_nodes = 1  # single-node per sample
    val_cb24 = make_val_callback_rolling(
        ds_train, scalers24, H24, aux24, label="24h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts24, df_truth_test=df_t24, zero_rmse_test=zero_rmse_24
    )
    mdl24 = train_model(X24, Y24, Din, H24, n_nodes, scalers24, eval_callback=val_cb24)

    # ==== 48h ====
    X48, Y48, _, scalers48, aux48 = prepare_training_split(ds_train, horizon=H48, train_last_day=TRAIN_LAST_DAY)
    val_cb48 = make_val_callback_rolling(
        ds_train, scalers48, H48, aux48, label="48h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts48, df_truth_test=df_t48, zero_rmse_test=zero_rmse_48
    )
    mdl48 = train_model(X48, Y48, Din, H48, n_nodes, scalers48, eval_callback=val_cb48)

    # ----- Test evaluation using best models, then write CSVs -----
    # 24h
    if mdl24 is not None:
        best24_path = os.path.join(OUT_DIR, "best_two_stage_24h.pt")
        if os.path.exists(best24_path):
            ckpt24 = torch.load(best24_path, map_location=DEVICE)
            try:
                mdl24.load_state_dict(ckpt24["model_state_dict"])
                rmse24_val = ckpt24.get("rmse_select", ckpt24.get("rmse", None))
                ep24 = ckpt24.get("epoch", None)
                if isinstance(rmse24_val, (float, int)):
                    print(f"[24h] Loaded best checkpoint (epoch {ep24}, Val selected RMSE={rmse24_val:.4f}, best_by={ckpt24.get('best_by','overall')})")
                else:
                    print(f"[24h] Loaded best checkpoint (epoch {ep24})")
            except Exception as e:
                print(f"[24h] Warning: failed to load best checkpoint: {e}")
        df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24, prob_thr=CLS_PROB_THR)
        test_rmse_24 = per_county_rmse_mean(df_t24, df24)
        # Ratios on test
        locs24 = [c for c in df_t24.columns if c != "timestamp"]
        df_zero_pred_24 = pd.DataFrame({
            "timestamp": np.repeat(df_t24["timestamp"].values, len(locs24)),
            "location": np.tile(locs24, len(df_t24)),
            "pred": 0.0
        })
        z2nz_24, nz2z_24 = zero_nonzero_ratios(df_t24, df24)
        z2nz_24_zero, nz2z_24_zero = zero_nonzero_ratios(df_t24, df_zero_pred_24)
        print(f"[Test 24h] Testing RMSE={test_rmse_24:.4f} (all-zero RMSE={zero_rmse_24:.4f})")
        print(f"[Test 24h] Zero→Nonzero={z2nz_24:.4f} (all-zero={z2nz_24_zero:.4f}) | Nonzero→Zero={nz2z_24:.4f} (all-zero={nz2z_24_zero:.4f})")
    else:
        df24 = None

    # 48h
    if mdl48 is not None:
        best48_path = os.path.join(OUT_DIR, "best_two_stage_48h.pt")
        if os.path.exists(best48_path):
            ckpt48 = torch.load(best48_path, map_location=DEVICE)
            try:
                mdl48.load_state_dict(ckpt48["model_state_dict"])
                rmse48_val = ckpt48.get("rmse_select", ckpt48.get("rmse", None))
                ep48 = ckpt48.get("epoch", None)
                if isinstance(rmse48_val, (float, int)):
                    print(f"[48h] Loaded best checkpoint (epoch {ep48}, Val selected RMSE={rmse48_val:.4f}, best_by={ckpt48.get('best_by','overall')})")
                else:
                    print(f"[48h] Loaded best checkpoint (epoch {ep48})")
            except Exception as e:
                print(f"[48h] Warning: failed to load best checkpoint: {e}")
        df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48, prob_thr=CLS_PROB_THR)
        test_rmse_48 = per_county_rmse_mean(df_t48, df48)
        # Ratios on test
        locs48 = [c for c in df_t48.columns if c != "timestamp"]
        df_zero_pred_48 = pd.DataFrame({
            "timestamp": np.repeat(df_t48["timestamp"].values, len(locs48)),
            "location": np.tile(locs48, len(df_t48)),
            "pred": 0.0
        })
        z2nz_48, nz2z_48 = zero_nonzero_ratios(df_t48, df48)
        z2nz_48_zero, nz2z_48_zero = zero_nonzero_ratios(df_t48, df_zero_pred_48)
        print(f"[Test 48h] Testing RMSE={test_rmse_48:.4f} (all-zero RMSE={zero_rmse_48:.4f})")
        print(f"[Test 48h] Zero→Nonzero={z2nz_48:.4f} (all-zero={z2nz_48_zero:.4f}) | Nonzero→Zero={nz2z_48:.4f} (all-zero={nz2z_48_zero:.4f})")
    else:
        df48 = None

    # Fallbacks
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

    out24 = os.path.join(OUT_DIR, "two_stage_pred_24h.csv")
    out48 = os.path.join(OUT_DIR, "two_stage_pred_48h.csv")
    df24.to_csv(out24, index=False)
    df48.to_csv(out48, index=False)

    print("Saved predictions:")
    print(f"  {out24}")
    print(f"  {out48}")

    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
