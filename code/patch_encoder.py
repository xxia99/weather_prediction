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
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-4
WEIGHT_DECAY  = 1e-2
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# === temporal dropout probability (drop whole timesteps during training) ===
TEMPORAL_DROP_PROB = 0.05  # e.g., 5% of timesteps zeroed per window

# === PatchTST encoder hyperparams ===
PATCH_LEN      = 2          # patch length along time (must be <= SEQ_LEN)
PATCH_STRIDE   = None       # None -> non-overlapping (stride = PATCH_LEN)
PATCH_D_MODEL  = 64         # token/patch embedding size
PATCH_NHEAD    = 4
PATCH_NLAYERS  = 2
PATCH_DROPOUT  = 0.3
PATCH_FF       = 4 * PATCH_D_MODEL  # transformer FFN width

# --------- StemGNN hyperparams (lightweight, tunable) ----------
STEM_D_MODEL  = 64     # feature hidden width
STEM_K_ORDER  = 2      # graph diffusion orders (0..K)
STEM_NBLOCKS  = 2      # stacked spatio-temporal blocks
STEM_D_EMB    = 32     # node embedding dim (for adaptive adjacency)
STEM_DROPOUT  = 0.5

# === Pretrained encoder options (NEW) ===
# PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst.pt" # e.g., "results_seq/pretrained_patchtst.pt" or None
# PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_multitask.pt" # e.g., "results_seq/pretrained_patchtst.pt" or None
# PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_multitask_contrast.pt"
PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_multitask_contrast_freq.pt"
ENCODER_LR              = 2e-5  # smaller LR for encoder fine-tuning
FREEZE_ENCODER          = True  # if True, keep encoder frozen during training

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
        self.mix = nn.Linear(d_model, d_model)

    def forward(self, x, A):
        # x: (B, T, N, C), A: (N, N)
        B, T, N, C = x.shape
        # Precompute powers of A
        A_pows = [torch.eye(N, device=x.device, dtype=x.dtype), A]
        for _ in range(2, self.k_order + 1):
            A_pows.append(A_pows[-1] @ A)
        # Diffusion sum_k alpha_k * (A^k X)
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
        # x: (B, T, N, F_in)
        h0 = self.in_proj(x)                 # (B, T, N, d_model)
        ht = self.tconv(h0)                  # temporal modeling
        A  = self.adaptA()                   # (N, N)
        hg = self.gdiff(ht, A)               # (B, T, N, d_model)
        h  = self.out_norm(h0 + hg)          # residual
        return h

# -------------------- StemGNN Forecaster (PatchTST encoder) --------------------
class STGNNForecaster(nn.Module):
    """
    PatchTST encoder -> stacked Stem blocks -> temporal pooling -> linear readout.
    Input:  x (B, T, N, F_in)
    Output: (B, N, H)
    """
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels

        # Patch encoder (per-node)
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

        # Optional: load pretrained encoder weights
        if PRETRAINED_ENCODER_PATH is not None and os.path.exists(PRETRAINED_ENCODER_PATH):
            try:
                ckpt = torch.load(PRETRAINED_ENCODER_PATH, map_location="cpu")
                state = ckpt.get("state_dict", ckpt)
                self.patch.load_state_dict(state, strict=False)
                print(f"[Encoder] Loaded pretrained weights from: {PRETRAINED_ENCODER_PATH}")
            except Exception as e:
                print(f"[Encoder] Warning: failed to load pretrained encoder: {e}")

        # Optionally freeze encoder
        if FREEZE_ENCODER:
            for p in self.patch.parameters():
                p.requires_grad = False
            print("[Encoder] Frozen (no fine-tuning).")

        # Stem blocks start from PATCH_D_MODEL
        blocks = []
        d_in = PATCH_D_MODEL
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

        # PatchTST encoder to (B, T, N, PATCH_D_MODEL)
        h = self.patch(x)

        # Spatio-temporal Stem blocks
        for blk in self.blocks:
            h = blk(h)              # (B, T, N, STEM_D_MODEL)

        # mean temporal pooling (more robust than last-step only)
        pooled = h.mean(dim=1)      # (B, N, STEM_D_MODEL)
        y = self.readout(pooled)    # (B, N, H)
        return y

# ---------------- training/infer (unchanged API except optimizer param groups) ----------------
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

    # Param groups: smaller LR for encoder (unless frozen)
    enc_params = list(model.patch.parameters())
    enc_ids = set(id(p) for p in enc_params)
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]

    if FREEZE_ENCODER:
        # Encoder params are frozen; optimize only decoder
        opt = torch.optim.Adam(other_params, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        opt = torch.optim.Adam(
            [
                {"params": enc_params,   "lr": ENCODER_LR},
                {"params": other_params, "lr": LR}
            ],
            weight_decay=WEIGHT_DECAY
        )

    loss_fn = nn.MSELoss()
    # loss_fn = nn.SmoothL1Loss(beta=1.0)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)  # xb: (B, T, N, F), yb: (B, N, H)

            # === Temporal Dropout — randomly zero a few timesteps per window ===
            if TEMPORAL_DROP_PROB > 0.0:
                with torch.no_grad():
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

# ---------------- pipeline helpers (unchanged) ----------------
def prepare_training(ds_train, horizon):
    """
    Returns:
      X: (num_windows, SEQ_LEN, N, Din)
      Y: (num_windows, N, horizon)
      Din = 1 (outage) + F (weather)
    """
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    T, N, F = w.shape

    # Scalers
    y_mu, y_sd = zfit(y.reshape(-1, 1))
    w_mu, w_sd = zfit(w.reshape(-1, F))

    # Scale
    y_sc = zapply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, N)         # (T, N)
    w_sc = zapply(w.reshape(-1, F), w_mu, w_sd).reshape(T, N, F)      # (T, N, F)

    Din = 1 + F
    X_list, Y_list = [], []
    for node in range(N):
        # Per-node feature timeline: concat scaled outage channel and weather channels
        x_loc_full = np.concatenate([y_sc[:, node:node+1], w_sc[:, node, :]], axis=1)  # (T, Din)
        Xw, Yw = build_windows(x_loc_full, y_sc[:, node], SEQ_LEN, horizon)            # (nW, SEQ_LEN, Din), (nW, H)
        if len(Xw):
            X_list.append(Xw)   # (nW, SEQ_LEN, Din)
            Y_list.append(Yw)   # (nW, H)

    # Stack across nodes → (nW, SEQ_LEN, N, Din) and (nW, N, H)
    X = np.stack(X_list, axis=2)  # (nW, SEQ_LEN, N, Din)
    Y = np.stack(Y_list, axis=2)  # (nW, H, N)
    Y = np.swapaxes(Y, 1, 2)      # -> (nW, N, H)

    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    return X, Y, Din, scalers

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

def make_eval_callback(ds_train, scalers, horizon, ts_future, df_truth, label):
    """Return a callback that evaluates the model after each epoch and saves the best one."""
    best = {"rmse": float("inf"), "epoch": -1}

    def _callback(model, epoch):
        df_pred = predict_per_county(ds_train, model, scalers, horizon, ts_future)
        rmse = per_county_rmse_mean(df_truth, df_pred)
        print(f"[Eval {label}] Epoch {epoch}: RMSE={rmse:.4f}")

        # Save best model so far
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
            print(f"[Eval {label}] New best RMSE {rmse:.4f} at epoch {epoch}. Saved: {save_path}")

    return _callback

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

    # Truth data for evaluation
    df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))
    # df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "valid_1day.nc"))
    # df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "valid_2days.nc"))

    # Train two models (24h, 48h)
    X24, Y24, Din, scalers24 = prepare_training(ds_train, horizon=H24)
    n_nodes = ds_train.sizes["location"]
    eval_cb24 = make_eval_callback(ds_train, scalers24, H24, ts24, df_t24, "24h")
    mdl24 = train_model(X24, Y24, Din, H24, n_nodes, eval_callback=eval_cb24)

    X48, Y48, _, scalers48 = prepare_training(ds_train, horizon=H48)
    eval_cb48 = make_eval_callback(ds_train, scalers48, H48, ts48, df_t48, "48h")
    mdl48 = train_model(X48, Y48, Din, H48, n_nodes, eval_callback=eval_cb48)

    # 24h
    if mdl24 is not None:
        best24_path = os.path.join(OUT_DIR, "best_stgnn_24h.pt")
        if os.path.exists(best24_path):
            ckpt24 = torch.load(best24_path, map_location=DEVICE)
            try:
                mdl24.load_state_dict(ckpt24["model_state_dict"])
                rmse24 = ckpt24.get("rmse", None)
                ep24 = ckpt24.get("epoch", None)
                if isinstance(rmse24, (float, int)):
                    print(f"[24h] Loaded best checkpoint (epoch {ep24}, RMSE={rmse24:.4f})")
                else:
                    print(f"[24h] Loaded best checkpoint (epoch {ep24})")
            except Exception as e:
                print(f"[24h] Warning: failed to load best checkpoint: {e}")
        df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24)
    else:
        df24 = None

    # 48h
    if mdl48 is not None:
        best48_path = os.path.join(OUT_DIR, "best_stgnn_48h.pt")
        if os.path.exists(best48_path):
            ckpt48 = torch.load(best48_path, map_location=DEVICE)
            try:
                mdl48.load_state_dict(ckpt48["model_state_dict"])
                rmse48 = ckpt48.get("rmse", None)
                ep48 = ckpt48.get("epoch", None)
                if isinstance(rmse48, (float, int)):
                    print(f"[48h] Loaded best checkpoint (epoch {ep48}, RMSE={rmse48:.4f})")
                else:
                    print(f"[48h] Loaded best checkpoint (epoch {ep48})")
            except Exception as e:
                print(f"[48h] Warning: failed to load best checkpoint: {e}")
        df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48)
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
    out24 = os.path.join(OUT_DIR, "stemgnn_pred_24h.csv")
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
