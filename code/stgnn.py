# demo_seq2seq_with_stgnn.py
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

# -------------- STGNN Imports --------------
# pip install torch-geometric-temporal
from torch_geometric_temporal.nn.recurrent import AGCRN

# ---------------- config ----------------
SPLIT_DIR = "data/"
OUT_DIR   = "results_seq/"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN       = 24
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM     = 32  # node embedding dims inside AGCRN (tunable)

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

# ---------------- STGNN (AGCRN) ----------------
class STGNNForecaster(nn.Module):
    """
    AGCRN expects a single time slice per call: X_t with shape (B, N, in_channels),
    plus node embeddings E: (N, embed_dim). We'll iterate over T inside forward:
      Input to forward: x (B, T, N, F)
      Loop t=0..T-1: out_t = agcrn(x[:, t], E) -> (B, N, H)
      Return last time step output: (B, N, H)
    """
    def __init__(self, in_channels, horizon, num_nodes, embed_dim=EMBED_DIM):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels

        # Learnable node embeddings E (N, embed_dim)
        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim))

        self.stgnn = AGCRN(
            number_of_nodes=num_nodes,
            in_channels=in_channels,
            out_channels=horizon,
            K=3,
            embedding_dimensions=embed_dim
        )

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        assert N == self.num_nodes and F == self.in_channels, \
            f"AGCRN input shape mismatch: got N={N},F={F}, expected N={self.num_nodes},F={self.in_channels}"

        last_out = None
        for t in range(T):
            x_t = x[:, t, :, :]                  # (B, N, F)
            out_t = self.stgnn(x_t, self.node_emb)  # (B, N, horizon)
            last_out = out_t
        return last_out                           # (B, N, horizon)

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

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)  # xb: (B, T, N, F), yb: (B, N, H)
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

# ---------------- pipeline helpers (STGNN-shaped) ----------------
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

# def make_eval_callback(ds_train, scalers, horizon, ts_future, df_truth, label):
#     def _callback(model, epoch):
#         df_pred = predict_per_county(ds_train, model, scalers, horizon, ts_future)
#         rmse = per_county_rmse_mean(df_truth, df_pred)
#         print(f"[Eval {label}] Epoch {epoch}: RMSE={rmse:.4f}")
#     return _callback
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
    n_nodes = ds_train.sizes["location"]  # avoids FutureWarning
    eval_cb24 = make_eval_callback(ds_train, scalers24, H24, ts24, df_t24, "24h")
    mdl24 = train_model(X24, Y24, Din, H24, n_nodes, eval_callback=eval_cb24)

    X48, Y48, _, scalers48 = prepare_training(ds_train, horizon=H48)
    eval_cb48 = make_eval_callback(ds_train, scalers48, H48, ts48, df_t48, "48h")
    mdl48 = train_model(X48, Y48, Din, H48, n_nodes, eval_callback=eval_cb48)

    # # Predict per county
    # df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24) if mdl24 is not None else None
    # df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48) if mdl48 is not None else None
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
    out24 = os.path.join(OUT_DIR, "stgnn_pred_24h.csv")
    out48 = os.path.join(OUT_DIR, "stgnn_pred_48h.csv")
    df24.to_csv(out24, index=False)
    df48.to_csv(out48, index=False)

    print("Saved predictions:")
    print(f"  {out24}")
    print(f"  {out48}")

    # Clean up
    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
