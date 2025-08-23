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
# PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_multitask.pt"
# PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_multitask_contrast.pt"
# PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_multitask_contrast_freq.pt"
PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_validation.pt"
ENCODER_LR              = 2e-5  # smaller LR for encoder fine-tuning
FREEZE_ENCODER          = True  # if True, keep encoder frozen during training

# === Use Chronos-Bolt PEFT encoder instead of PatchTST (NEW) ===
USE_CB_ENCODER          = True
# CB_ENCODER_CKPT         = "results_seq/pretrained_cb_encoder_peft_aug.pt"   # "results_seq/pretrained_cb_encoder_peft_multitask_contrast_freq.pt"
# CB_ENCODER_CKPT         = "results_seq/cb_peft_adapter_tsmix.pt"   # "results_seq/pretrained_cb_encoder_peft_multitask_contrast_freq.pt"
# CB_ENCODER_CKPT         = "results_seq/pretrained_cb_encoder_peft_aug_100best.pt"
CB_ENCODER_CKPT         = "results_seq/pretrained_cb_encoder_peft_aug_H48.pt"

# === Day split for in-train validation (NEW) ===
TRAIN_LAST_DAY = 80   # use days 1..80 for training; 81..end for validation
LOSS_HUBER_BETA = 1.0  # Huber delta

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

# -------------------- Chronos-Bolt PEFT encoder (NEW) --------------------
# Only used when USE_CB_ENCODER=True, otherwise ignored.
def _discover_lora_targets(model: nn.Module):
    """Find Linear submodule name fragments to target with LoRA; robust to model variants."""
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    candidates = [
        "q_proj","k_proj","v_proj","o_proj",
        "query_key_value","out_proj","in_proj",
        "to_q","to_k","to_v","to_out","proj_in","proj_out",
        "q","k","v","o","wi","wo","wi_0","wi_1",
        "Wqkv","Wo","Wq","Wk","Wv",
    ]
    present = [s for s in candidates if any(s in n for n in linear_names)]
    if not present:
        attn_linear = [n for n in linear_names if ("attn" in n.lower() or "attention" in n.lower())]
        if attn_linear:
            present = list({n.split(".")[-1] for n in attn_linear})
    if not present:
        enc_linear = [n for n in linear_names if (".encoder." in n or "encoder.layers" in n or ".layers." in n)]
        if enc_linear:
            present = list({n.split(".")[-1] for n in enc_linear})
    if not present:
        present = ["linear"]
    # unique order-preserving
    seen, final = set(), []
    for s in present:
        if s not in seen:
            seen.add(s)
            final.append(s)
    return final

class ChronosBoltEncoderPEFT(nn.Module):
    """
    Loads the LoRA-tuned Chronos-Bolt encoder checkpoint produced by your PEFT pretrain script,
    and exposes the SAME interface as PatchTSTEncoder:
        forward(x: (B,T,N,F)) -> (B,T,N,D_OUT)
    """
    def __init__(self, in_channels, ckpt_path):
        super().__init__()
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"CB encoder ckpt not found: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu")
        cb_model_name = ck.get("cb_model_name", "amazon/chronos-bolt-mini")
        d_out         = int(ck.get("d_out", 64))
        self.in_channels = int(ck.get("in_channels", in_channels))
        self.d_model = d_out  # for compatibility with downstream code expecting .d_model

        # lazy imports to avoid dependency if not used
        try:
            from transformers import AutoModel, AutoConfig
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise ImportError(
                "This script requires 'transformers' and 'peft' when USE_CB_ENCODER=True.\n"
                "Install: pip install transformers peft accelerate safetensors\n"
                f"Original import error: {e}"
            )

        config  = AutoConfig.from_pretrained(cb_model_name, trust_remote_code=True)
        backbone = AutoModel.from_pretrained(cb_model_name, trust_remote_code=True, output_hidden_states=True)

        lcfg = LoraConfig(
            r=int(ck.get("lora_r", 8)),
            lora_alpha=int(ck.get("lora_alpha", 16)),
            lora_dropout=float(ck.get("lora_dropout", 0.05)),
            target_modules=_discover_lora_targets(backbone),
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        backbone = get_peft_model(backbone, lcfg)

        # Small IO adapters (same shapes as in pretraining)
        hidden_dim = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if hidden_dim is None:
            raise RuntimeError("Cannot determine Chronos hidden size from config.")
        self.embed_proj = nn.Linear(self.in_channels, hidden_dim)
        self.post_proj  = nn.Linear(hidden_dim, self.d_model)

        # Load full wrapper state dict
        self.backbone = backbone
        self.load_state_dict(ck["state_dict"], strict=True)
        print(f"[CB Encoder] Loaded PEFT encoder from: {ckpt_path} | d_out={self.d_model}")

    def forward(self, x):
        # x: (B,T,N,F) scaled
        B, T, N, F = x.shape
        assert F == self.in_channels, f"CB encoder expected F={self.in_channels}, got {F}"

        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)      # (B*N, T, F)
        emb  = self.embed_proj(x_bn)                           # (B*N, T, H)
        attn_mask = torch.ones(B * N, T, dtype=torch.long, device=emb.device)

        # T5-like: use encoder(inputs_embeds=...)
        if hasattr(self.backbone, "encoder") and callable(self.backbone.encoder):
            out = self.backbone.encoder(
                inputs_embeds=emb, attention_mask=attn_mask, output_hidden_states=True
            )
            h_bn = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        else:
            out = self.backbone(
                inputs_embeds=emb, attention_mask=attn_mask, output_hidden_states=True
            )
            h_bn = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        # time alignment
        if h_bn.size(1) > T:
            h_bn = h_bn[:, :T, :]
        elif h_bn.size(1) < T:
            pad_len = T - h_bn.size(1)
            pad = torch.zeros(h_bn.size(0), pad_len, h_bn.size(2), device=h_bn.device, dtype=h_bn.dtype)
            h_bn = torch.cat([h_bn, pad], dim=1)

        h_bn = self.post_proj(h_bn)                            # (B*N, T, d_model)
        h = h_bn.view(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B, T, N, d_model)
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

# -------------------- StemGNN Forecaster (PatchTST or CB encoder) --------------------
class STGNNForecaster(nn.Module):
    """
    Encoder -> stacked Stem blocks -> temporal pooling -> linear readout.
    Input:  x (B, T, N, F_in)
    Output: (B, N, H)
    """
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels

        # Encoder choice
        if USE_CB_ENCODER and os.path.exists(CB_ENCODER_CKPT):
            self.patch = ChronosBoltEncoderPEFT(in_channels=in_channels, ckpt_path=CB_ENCODER_CKPT)
            print("[Encoder] Using Chronos-Bolt PEFT encoder.")
        else:
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
            # Optional: load pretrained PatchTST encoder weights
            if PRETRAINED_ENCODER_PATH is not None and os.path.exists(PRETRAINED_ENCODER_PATH):
                try:
                    ckpt = torch.load(PRETRAINED_ENCODER_PATH, map_location="cpu")
                    state = ckpt.get("state_dict", ckpt)
                    self.patch.load_state_dict(state, strict=False)
                    print(f"[Encoder] Loaded pretrained PatchTST from: {PRETRAINED_ENCODER_PATH}")
                except Exception as e:
                    print(f"[Encoder] Warning: failed to load pretrained PatchTST: {e}")

        # Optionally freeze encoder
        if FREEZE_ENCODER:
            for p in self.patch.parameters():
                p.requires_grad = False
            print("[Encoder] Frozen (no fine-tuning).")

        # Stem blocks start from actual encoder output dim
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
        self.readout = nn.Linear(STEM_D_MODEL, horizon)

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        assert N == self.num_nodes and F == self.in_channels, \
            f"StemGNN input shape mismatch: got N={N},F={F}, expected N={self.num_nodes},F={self.in_channels}"

        # Encoder to (B, T, N, enc_out_dim)
        h = self.patch(x)

        # Spatio-temporal Stem blocks
        for blk in self.blocks:
            h = blk(h)              # (B, T, N, STEM_D_MODEL)

        # mean temporal pooling (more robust than last-step only)
        pooled = h.mean(dim=1)      # (B, N, STEM_D_MODEL)
        y = self.readout(pooled)    # (B, N, H)
        return y

# ---------------- helpers for day split & windows (NEW) ----------------
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

# ---------------- validation roll (NEW) ----------------
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

# -------- helper for all-zero baseline on a given truth frame (NEW) --------
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

def make_val_callback_rolling(ds_train, scalers, horizon, aux, label,
                              train_last_day=80,
                              ts_future_test=None, df_truth_test=None, zero_rmse_test=float("nan")):
    """
    Evaluate each epoch on rolling day-ahead validation (days 81..end) **and** on the test set.
    Prints:
      - Rolling RMSE=xxx.xxxx (all-zero RMSE=yyy.yyyy)
      - Testing RMSE=aaa.aaaa (all-zero RMSE=bbb.bbbb)
    Saves best checkpoint (by validation RMSE) like before.
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
            # still compute test if requested
        else:
            rmse = per_county_rmse_mean(df_truth_val, df_pred_val)
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE={rmse:.4f} (all-zero RMSE={zero_rmse_val:.4f})")

            # Save best by validation RMSE
            if np.isfinite(rmse) and rmse < best["rmse"]:
                best["rmse"] = rmse
                best["epoch"] = epoch
                save_path = os.path.join(OUT_DIR, f"best_enc-dec_{label}.pt")
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

        # ---- Test set evaluation at each validation step (NEW) ----
        if ts_future_test is not None and df_truth_test is not None and len(ts_future_test) > 0:
            df_pred_test = predict_per_county(ds_train, model, scalers, horizon, ts_future_test)
            rmse_test = per_county_rmse_mean(df_truth_test, df_pred_test)
            print(f"[Val {label}] Epoch {epoch}: Testing RMSE={rmse_test:.4f} (all-zero RMSE={zero_rmse_test:.4f})")

    return _callback

# ---------------- training/infer (unchanged API except eval callback) ----------------
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
    loss_fn = nn.SmoothL1Loss(beta=LOSS_HUBER_BETA)

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

# ---------------- pipeline helpers (train windows only) ----------------
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

    # Scale inputs with train scalers
    y_sc = zapply(y.reshape(-1, 1), scalers["y_mu"], scalers["y_sd"]).reshape(T, N)    # (T, N)
    w_sc = zapply(w.reshape(-1, F), scalers["w_mu"], scalers["w_sd"]).reshape(T, N, F) # (T, N, F)

    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)   # (T, N, Din)
    Xin = X_all[-SEQ_LEN:]                                    # (SEQ_LEN, N, Din)
    Xin = Xin.reshape(1, SEQ_LEN, N, Din)                     # (1, SEQ_LEN, N, Din)

    pred_sc = infer(model, Xin)[0]                            # (N, H)

    pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
    pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

    rows = []
    for i, county in enumerate(locs):
        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred[i].astype(float)}))
    return pd.concat(rows, ignore_index=True)

def make_eval_callback(ds_train, scalers, horizon, ts_future, df_truth, label):
    """Original test callback (kept)."""
    best = {"rmse": float("inf"), "epoch": -1}

    def _callback(model, epoch):
        df_pred = predict_per_county(ds_train, model, scalers, horizon, ts_future)
        rmse = per_county_rmse_mean(df_truth, df_pred)
        print(f"[Eval {label}] Epoch {epoch}: RMSE={rmse:.4f}")

        if np.isfinite(rmse) and rmse < best["rmse"]:
            best["rmse"] = rmse
            best["epoch"] = epoch
            save_path = os.path.join(OUT_DIR, f"best_enc-dec_{label}.pt")
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

    # Horizons and timestamps (for test eval; unchanged)
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    # Truth data for test (wide)
    df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))
    # df_t24 = load_truth_wide(os.path.join(SPLIT_DIR, "valid_1day.nc"))
    # df_t48 = load_truth_wide(os.path.join(SPLIT_DIR, "valid_2days.nc"))

    # Pre-compute all-zero RMSE for test sets (NEW)
    zero_rmse_24 = _all_zero_rmse_for_truth(df_t24)
    zero_rmse_48 = _all_zero_rmse_for_truth(df_t48)

    # Train two models (24h, 48h) with split-aware training and rolling validation + test-at-val
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

    # ----- Test evaluation using best models, then write CSVs (keep previous behavior) -----
    # 24h
    if mdl24 is not None:
        best24_path = os.path.join(OUT_DIR, "best_enc-dec_24h.pt")
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
        # Compute test RMSE vs truth and print with all-zero baseline
        test_rmse_24 = per_county_rmse_mean(df_t24, df24)
        print(f"[Test 24h] Testing RMSE={test_rmse_24:.4f} (all-zero RMSE={zero_rmse_24:.4f})")
    else:
        df24 = None

    # 48h
    if mdl48 is not None:
        best48_path = os.path.join(OUT_DIR, "best_enc-dec_48h.pt")
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
        # Compute test RMSE vs truth and print with all-zero baseline
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

    # Save test predictions (unchanged; now after the printed RMSEs)
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
