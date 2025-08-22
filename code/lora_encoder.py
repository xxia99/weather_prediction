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

# === PatchTST (fallback) encoder hyperparams ===
PATCH_LEN      = 2          # patch length along time (must be <= SEQ_LEN)
PATCH_STRIDE   = None       # None -> non-overlapping (stride = PATCH_LEN)
PATCH_D_MODEL  = 64         # token/patch embedding size
PATCH_NHEAD    = 4
PATCH_NLAYERS  = 2
PATCH_DROPOUT  = 0.3
PATCH_FF       = 4 * PATCH_D_MODEL  # transformer FFN width

# --------- StemGNN hyperparams (lightweight, tunable) ----------
STEM_D_MODEL  = 64     # feature hidden width (decoder expects encoder to output this via adapter)
STEM_K_ORDER  = 2      # graph diffusion orders (0..K)
STEM_NBLOCKS  = 2      # stacked spatio-temporal blocks
STEM_D_EMB    = 32     # node embedding dim (for adaptive adjacency)
STEM_DROPOUT  = 0.5

# === Pretrained encoder options (kept for PatchTST fallback) ===
PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_multitask_contrast_freq.pt"
ENCODER_LR              = 2e-5   # smaller LR for encoder or adapters
FREEZE_ENCODER          = True   # freeze base encoder; adapters remain trainable

# === Residual Calibrator hyperparams (kept) ===
CAL_EPOCHS        = 6
CAL_LR            = 2e-3
CAL_WEIGHT_DECAY  = 1e-4
CAL_HIDDEN        = 128
CAL_DROPOUT       = 0.1

# === Chronos-Bolt options (NEW encoder) ===
USE_CB_ENCODER = True                # try to use Chronos-Bolt as encoder; fallback to PatchTST if unavailable
CB_MODEL_NAME  = "/home2/caojia/weather_prediction/code/chronos-bolt-small"  # HF path or local dir
CB_EMB_DIM     = 64                   # length of CB prediction used as embedding token (PEFT dimension)
CB_BATCHED     = False                # if True and pipeline supports batching, you can adapt predict loop

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

# -------------------- PatchTST fallback encoder --------------------
class PatchTSTEncoder(nn.Module):
    """
    Channel-last patching along time. For each node independently:
      Input  : (B, T, N, F)
      Output : (B, T, N, d_model)
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
        L = (T + S - 1) // S
        T_pad = (L - 1) * S + P
        if T_pad < T:
            L += 1
            T_pad = (L - 1) * S + P
        if T_pad > T:
            pad = torch.zeros(B * N, T_pad - T, F, device=x.device, dtype=x.dtype)
            x_bn = torch.cat([x_bn, pad], dim=1)  # (B*N, T_pad, F)

        # patches
        x_unf = x_bn.unfold(dimension=1, size=P, step=S)          # (B*N, L, P, F)
        x_unf = x_unf.contiguous().view(B * N, L, P * F)          # (B*N, L, P*F)

        # token embedding + transformer
        tokens = self.proj(x_unf)                                 # (B*N, L, d_model)
        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)                              # (B*N, L, d_model)

        # unpatch (assume S==P) -> repeat per P and trim to T
        tokens_rep = tokens.repeat_interleave(P, dim=1)            # (B*N, L*P, d_model)
        steps = tokens_rep[:, :T, :]                               # (B*N, T, d_model)

        # reshape back
        h = steps.view(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B, T, N, d_model)
        return h

def _sanitize_chronos_config(model_dir, keys_to_drop=("input_patch_size",)):
    """
    Copy model_dir to a temp folder and remove unknown config keys so older Chronos
    packages can load it. Returns the sanitized directory path.
    """
    if not os.path.isdir(model_dir):
        # If it's not a local dir, we can't safely edit it.
        return model_dir
    tmp_out = os.path.join(tempfile.gettempdir(), "cb_sanitized")
    # fresh copy
    if os.path.exists(tmp_out):
        shutil.rmtree(tmp_out)
    shutil.copytree(model_dir, tmp_out)
    cfg_path = os.path.join(tmp_out, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            changed = False
            for k in keys_to_drop:
                if k in cfg:
                    cfg.pop(k, None)
                    changed = True
            if changed:
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"[CB-Encoder] Sanitized config removed keys: {keys_to_drop}")
        except Exception as e:
            print(f"[CB-Encoder] Warning: failed to sanitize config: {e}")
    return tmp_out

# -------------------- Chronos-Bolt PEFT Encoder (NEW) --------------------
class ChronosBoltEncoderPEFT(nn.Module):
    """
    Uses Chronos-Bolt predictions as fixed-length embeddings (length=CB_EMB_DIM).
    - Chronos-Bolt backbone stays frozen (parameter-efficient).
    - Trainable adapters project CB embeddings -> STEM_D_MODEL for the decoder.
    - If Chronos is unavailable, falls back to PatchTSTEncoder automatically.

    Input : (B, T, N, F)  [scaled space]
    Output: (B, T, N, STEM_D_MODEL)
    """
    def __init__(self, in_channels, out_d_model=STEM_D_MODEL):
        super().__init__()
        self.in_channels = in_channels
        self.out_d_model = out_d_model

        self._cb_available = False
        self._cb = None

        if USE_CB_ENCODER:
            try:
                from chronos import ChronosPipeline  # type: ignore
                self._cb = ChronosPipeline.from_pretrained(CB_MODEL_NAME, device_map="auto")
                # freeze any internal params
                for p in self._cb.parameters() if hasattr(self._cb, "parameters") else []:
                    p.requires_grad = False
                self._cb_available = True
                print(f"[CB-Encoder] Loaded Chronos-Bolt from '{CB_MODEL_NAME}'. Backbone frozen.")
            except Exception as e:
                print(f"[CB-Encoder] Warning: Chronos-Bolt unavailable ({e}). Falling back to PatchTST.")

        if not self._cb_available:
            # Fallback encoder
            self.fallback = PatchTSTEncoder(
                in_channels=in_channels,
                d_model=PATCH_D_MODEL,
                nhead=PATCH_NHEAD,
                nlayers=PATCH_NLAYERS,
                dropout=PATCH_DROPOUT,
                ff=PATCH_FF,
                patch_len=PATCH_LEN,
                stride=PATCH_STRIDE,
            )
            # optional: load PatchTST pretrained weights
            if PRETRAINED_ENCODER_PATH is not None and os.path.exists(PRETRAINED_ENCODER_PATH):
                try:
                    ckpt = torch.load(PRETRAINED_ENCODER_PATH, map_location="cpu")
                    state = ckpt.get("state_dict", ckpt)
                    self.fallback.load_state_dict(state, strict=False)
                    print(f"[Fallback] Loaded PatchTST weights from: {PRETRAINED_ENCODER_PATH}")
                except Exception as e:
                    print(f"[Fallback] Could not load PatchTST weights: {e}")
            # projection to decoder width
            self.post = nn.Linear(PATCH_D_MODEL, out_d_model)
        else:
            # small adapters on CB embeddings (project CB_EMB_DIM -> out_d_model)
            self.pre  = nn.Linear(self.in_channels, self.in_channels, bias=False)  # optional tiny pre-adapter on features
            self.act  = nn.GELU()
            self.post = nn.Linear(CB_EMB_DIM, out_d_model)
            # by default, all adapter params are trainable. CB backbone is frozen.

        # honor FREEZE_ENCODER by freezing backbone (already frozen) + adapters if requested
        if FREEZE_ENCODER:
            self.freeze_backbone_and_adapters()

    def freeze_backbone_and_adapters(self):
        # freeze everything inside this encoder EXCEPT keep compatible with "parameter-efficient" desire
        # If you still want adapters trainable, comment out the loop below.
        for p in self.parameters():
            p.requires_grad = False
        # Keep adapters trainable (parameter-efficient fine-tuning)
        if self._cb_available:
            for p in self.pre.parameters():  p.requires_grad = True
            for p in self.post.parameters(): p.requires_grad = True
        else:
            for p in self.post.parameters(): p.requires_grad = True
            # keep PatchTST fallback frozen if FREEZE_ENCODER=True

    def trainable_parameters(self):
        # Return only parameters we want to optimize (adapters / fallback post-proj)
        if self._cb_available:
            params = list(self.pre.parameters()) + list(self.post.parameters())
        else:
            params = list(self.post.parameters())
            # If you prefer to fine-tune fallback PatchTST when not frozen:
            if not FREEZE_ENCODER:
                params += list(self.fallback.parameters())
        return params

    @torch.no_grad()
    def _cb_predict_vector(self, series_1d):
        """
        series_1d: numpy array (T,) [scaled]. We treat it as generic input to CB.
        We request CB to forecast CB_EMB_DIM steps; use the mean as embedding.
        Falls back to seasonal naive if CB/predict fails.
        """
        try:
            out = self._cb.predict(context=np.asarray(series_1d, dtype=np.float32),
                                   prediction_length=CB_EMB_DIM, num_samples=1)
            if isinstance(out, dict):
                if "mean" in out:
                    emb = np.asarray(out["mean"], dtype=np.float32)
                elif "samples" in out:
                    emb = np.asarray(out["samples"], dtype=np.float32).mean(axis=0)
                else:
                    emb = np.asarray(out, dtype=np.float32)
            else:
                emb = np.asarray(out, dtype=np.float32)
            emb = emb.reshape(-1)[:CB_EMB_DIM]
            if emb.shape[0] < CB_EMB_DIM:
                emb = np.pad(emb, (0, CB_EMB_DIM - emb.shape[0]), mode="wrap")
            return emb.astype(np.float32)
        except Exception:
            # Seasonal naive fallback
            period = min(24, max(1, len(series_1d)))
            tail = series_1d[-period:] if len(series_1d) >= period else np.resize(series_1d, period)
            reps = int(np.ceil(CB_EMB_DIM / period))
            emb = np.tile(tail, reps)[:CB_EMB_DIM].astype(np.float32)
            return emb

    def forward(self, x):
        """
        x: (B, T, N, F) -> h: (B, T, N, out_d_model)
        CB path: builds an embedding vector per (B,N) from outage channel via CB,
                 then tiles it across T (parameter-efficient and cheap).
        """
        B, T, N, F = x.shape
        if not self._cb_available:
            h = self.fallback(x)            # (B, T, N, PATCH_D_MODEL)
            h = self.post(h)                # (B, T, N, out_d_model)
            return h

        # tiny pre-adapter on features (learnable)
        x = self.pre(x)
        x = self.act(x)

        # Use only the outage channel (feature 0) to query CB
        x_np = x[..., 0].detach().cpu().numpy()  # (B, T, N)

        # Build embeddings per (b, n)
        embs = np.zeros((B, N, CB_EMB_DIM), dtype=np.float32)
        for b in range(B):
            for n in range(N):
                embs[b, n, :] = self._cb_predict_vector(x_np[b, :, n])

        # Convert to tensor and expand over time steps
        embs_t = torch.tensor(embs, device=x.device, dtype=x.dtype)      # (B, N, CB_EMB_DIM)
        embs_t = embs_t.unsqueeze(1).expand(B, T, N, CB_EMB_DIM)         # (B, T, N, CB_EMB_DIM)
        h = self.post(embs_t)                                            # (B, T, N, out_d_model)
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

# -------------------- StemGNN Forecaster (Chronos-Bolt encoder) --------------------
class STGNNForecaster(nn.Module):
    """
    Chronos-Bolt PEFT encoder (or PatchTST fallback) -> stacked Stem blocks -> temporal pooling -> linear readout.
    Input:  x (B, T, N, F_in)
    Output: (B, N, H)
    """
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels

        # NEW: Chronos-Bolt encoder with adapters; fallback to PatchTST transparently
        self.encoder = ChronosBoltEncoderPEFT(in_channels=in_channels)

        # Stem blocks start from STEM_D_MODEL (encoder already outputs this width)
        blocks = []
        d_in = STEM_D_MODEL
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

        # Encoder to (B, T, N, STEM_D_MODEL)
        h = self.encoder(x)

        # Spatio-temporal Stem blocks
        for blk in self.blocks:
            h = blk(h)              # (B, T, N, STEM_D_MODEL)

        # mean temporal pooling (robust)
        pooled = h.mean(dim=1)      # (B, N, STEM_D_MODEL)
        y = self.readout(pooled)    # (B, N, H)
        return y

# ---------------- training/infer (param groups updated to encoder) ----------------
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

    # Param groups: encoder adapters (PEFT) get ENCODER_LR; decoder gets LR
    enc_params = list(model.encoder.trainable_parameters())
    enc_ids = set(id(p) for p in enc_params)
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]

    opt = torch.optim.Adam(
        [
            {"params": enc_params,   "lr": ENCODER_LR},
            {"params": other_params, "lr": LR}
        ],
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.MSELoss()

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

# ====================== Residual Calibrator (kept) ======================
class ResidualCalibrator(nn.Module):
    """
    Per-node MLP that maps concat([baseline, stgnn], dim=-1) -> residual to add to baseline.
    Operates in SCALED space (same as training).
    """
    def __init__(self, horizon, hidden=CAL_HIDDEN, dropout=CAL_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * horizon, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x_cat):
        # x_cat: (B, N, 2H)
        B, N, D = x_cat.shape
        h = x_cat.reshape(B * N, D)
        r = self.net(h)                    # (B*N, H)
        return r.view(B, N, -1)            # (B, N, H)

def _seasonal_naive(y_hist, H, period=24):
    """Repeat last 'period' steps to fill next H. y_hist: (T,) unscaled."""
    if period <= 0 or len(y_hist) == 0:
        return np.repeat(y_hist[-1] if len(y_hist) else 0.0, H)
    tail = y_hist[-period:] if len(y_hist) >= period else np.resize(y_hist, period)
    reps = int(np.ceil(H / period))
    return np.tile(tail, reps)[:H]

def _cb_predict_one_series(y_hist, H):
    """
    Chronos-Bolt single-series forecast with seasonal-naive fallback.
    """
    try:
        if not USE_CB_ENCODER:
            raise ImportError("CB disabled by flag")
        from chronos import ChronosPipeline  # lazy import
        pipe = ChronosPipeline.from_pretrained(CB_MODEL_NAME, device_map="auto")
        ctx = np.asarray(y_hist, dtype=np.float32)
        out = pipe.predict(context=ctx, prediction_length=H, num_samples=1)
        if isinstance(out, dict):
            if "mean" in out:
                pred = np.asarray(out["mean"], dtype=np.float32)
            elif "samples" in out:
                pred = np.asarray(out["samples"], dtype=np.float32).mean(axis=0)
            else:
                pred = np.asarray(out, dtype=np.float32)
        else:
            pred = np.asarray(out, dtype=np.float32)
        return pred.reshape(-1)[:H]
    except Exception:
        return _seasonal_naive(np.asarray(y_hist, dtype=np.float32), H)

def cb_baseline_from_x_scaled(xb_scaled, horizon, scalers):
    """
    xb_scaled: torch.Tensor (B, T, N, Din) in scaled space.
    Return: torch.Tensor baseline in scaled space (B, N, H) using Chronos-Bolt (or fallback).
    """
    assert isinstance(xb_scaled, torch.Tensor)
    y_mu = float(scalers["y_mu"].flatten()[0])
    y_sd = float(scalers["y_sd"].flatten()[0]) if float(scalers["y_sd"].flatten()[0]) != 0.0 else 1.0

    B, T, N, Din = xb_scaled.shape
    y_hist_unscaled = (xb_scaled[..., 0].detach().cpu().numpy() * y_sd) + y_mu  # (B, T, N)

    base = np.zeros((B, N, horizon), dtype=np.float32)
    for b in range(B):
        for n in range(N):
            base[b, n, :] = _cb_predict_one_series(y_hist_unscaled[b, :, n], horizon)

    base_scaled = (base - y_mu) / (y_sd if y_sd != 0 else 1.0)
    return torch.tensor(base_scaled, dtype=torch.float32, device=xb_scaled.device)  # (B, N, H)

def train_residual_calibrator(X, Y, stgnn_model, horizon, label, scalers):
    """
    Train residual calibrator on the same windows used for STGNN training.
    Baseline = Chronos-Bolt (or seasonal-naive fallback). Works in scaled space.
    Saves to OUT_DIR/residual_calibrator_{label}.pt
    """
    if len(X) == 0:
        return None

    stgnn_model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    calib = ResidualCalibrator(horizon=horizon).to(DEVICE)
    opt = torch.optim.Adam(calib.parameters(), lr=CAL_LR, weight_decay=CAL_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    for epoch in range(CAL_EPOCHS):
        calib.train()
        ep_loss = 0.0
        start = time.time()
        for xb, yb in tqdm(dl, desc=f"[Calib {label}] Epoch {epoch+1}/{CAL_EPOCHS}", leave=False):
            xb = xb.to(DEVICE)     # (B, T, N, Din) scaled
            yb = yb.to(DEVICE)     # (B, N, H) scaled

            with torch.no_grad():
                y_stgnn = stgnn_model(xb)                                # (B, N, H) scaled
                y_base  = cb_baseline_from_x_scaled(xb, horizon, scalers)  # (B, N, H) scaled

            x_cat = torch.cat([y_base, y_stgnn], dim=-1)    # (B, N, 2H)
            opt.zero_grad()
            r = calib(x_cat)                                # (B, N, H)
            y_hat = y_base + r                              # (B, N, H)
            loss = loss_fn(y_hat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(calib.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item() * xb.size(0)

        ep_loss /= len(dl.dataset)
        print(f"[Calib {label}] Epoch {epoch+1}: Loss={ep_loss:.6f} | Time={time.time()-start:.2f}s")

    save_path = os.path.join(OUT_DIR, f"residual_calibrator_{label}.pt")
    torch.save(
        {
            "horizon": horizon,
            "state_dict": calib.state_dict(),
            "cal_hidden": CAL_HIDDEN,
            "cal_dropout": CAL_DROPOUT,
        },
        save_path,
    )
    print(f"[Calib {label}] Saved calibrator to {save_path}")
    return save_path

# ---------------- prediction (unchanged, uses calibrator if present) ----------------
def predict_per_county(ds_train, model, scalers, horizon, ts_future):
    """
    Predict next 'horizon' for ALL counties at once using last SEQ_LEN hours of train.
    If a residual calibrator checkpoint exists for this horizon, apply:
        final = baseline + Calib(concat(baseline, stgnn))
    Else fall back to STGNN predictions.
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

    # STGNN prediction (scaled)
    stgnn_sc = infer(model, Xin)[0]                           # (N, H)

    # Try residual calibrator
    calib_path_candidates = [
        os.path.join(OUT_DIR, "residual_calibrator_24h.pt"),
        os.path.join(OUT_DIR, "residual_calibrator_48h.pt"),
    ]
    calib_ckpt = None
    for pth in calib_path_candidates:
        if os.path.exists(pth):
            ck = torch.load(pth, map_location=DEVICE)
            if int(ck.get("horizon", -1)) == int(horizon):
                calib_ckpt = ck
                break

    if calib_ckpt is not None:
        # Chronos-Bolt baseline (scaled) on the same last SEQ_LEN window
        Xin_t = torch.tensor(Xin, dtype=torch.float32, device=DEVICE)  # (1, T, N, Din)
        y_base_sc = cb_baseline_from_x_scaled(Xin_t, horizon, scalers)[0]  # (N, H)

        # Prepare calibrator and apply
        calib = ResidualCalibrator(
            horizon=horizon,
            hidden=calib_ckpt.get("cal_hidden", CAL_HIDDEN),
            dropout=calib_ckpt.get("cal_dropout", CAL_DROPOUT),
        ).to(DEVICE)
        calib.load_state_dict(calib_ckpt["state_dict"])
        calib.eval()

        with torch.no_grad():
            x_cat = torch.cat(
                [y_base_sc.unsqueeze(0), torch.tensor(stgnn_sc, device=DEVICE).unsqueeze(0)],
                dim=-1,
            )  # (1, N, 2H)
            r_sc = calib(x_cat)[0].cpu().numpy()            # (N, H)
            pred_sc = y_base_sc.cpu().numpy() + r_sc        # (N, H)
    else:
        # No calibrator -> use STGNN directly (scaled)
        pred_sc = stgnn_sc

    # Invert scaling to original space
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

    # === Train residual calibrators on the same windows (kept) ===
    if mdl24 is not None and len(X24) > 0:
        train_residual_calibrator(X24, Y24, mdl24, horizon=H24, label="24h", scalers=scalers24)
    if mdl48 is not None and len(X48) > 0:
        train_residual_calibrator(X48, Y48, mdl48, horizon=H48, label="48h", scalers=scalers48)

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
