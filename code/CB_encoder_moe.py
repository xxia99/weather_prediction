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
import matplotlib.pyplot as plt
from demo_evaluate import load_truth_wide, per_county_rmse_mean
# (we no longer call event_stratified_report to avoid duplicate prints)

# ---------------- config ----------------
SPLIT_DIR = "data/"
# OUT_DIR   = "results_seq_mse_pre-CB_use-moe/update_restore_best"
OUT_DIR   = "results_seq_mse_pre-CB_use-moe/update_new_lora_2"
# OUT_DIR   = "results_seq_mse_pre-patch_use-moe"
# OUT_DIR   = "results_seq_mse_non-pre-patch_use-moe/2experts"
os.makedirs(OUT_DIR, exist_ok=True)

# core training hyperparameters
SEQ_LEN       = 24
BATCH_SIZE    = 64        # larger batch for more stable gradients
EPOCHS        = 40        # train for more epochs
LR            = 5e-4      # higher LR for faster convergence
WEIGHT_DECAY  = 5e-5      # lighter L2 regularization
DEVICE        = "cuda:1" if torch.cuda.is_available() else "cpu"

# === temporal dropout probability (drop whole timesteps during training) ===
# disable aggressive temporal dropout; rely on standard dropout instead
TEMPORAL_DROP_PROB = 0.0

# === PatchTST encoder hyperparams ===
PATCH_LEN      = 2          # patch length along time (must be <= SEQ_LEN)
PATCH_STRIDE   = None       # None -> non-overlapping (stride = PATCH_LEN)
PATCH_D_MODEL  = 64         # token/patch embedding size
PATCH_NHEAD    = 4
PATCH_NLAYERS  = 2
PATCH_DROPOUT  = 0.05     # reduce transformer dropout
PATCH_FF       = 4 * PATCH_D_MODEL  # transformer FFN width

# --------- StemGNN hyperparams (lightweight, tunable) ----------
STEM_D_MODEL  = 64     # feature hidden width
STEM_K_ORDER  = 2      # enable 2-hop spatial diffusion
STEM_NBLOCKS  = 2      # stacked spatio-temporal blocks
STEM_D_EMB    = 32     # node embedding dim (for adaptive adjacency)
STEM_DROPOUT  = 0.03   # lighter dropout

# === Pretrained encoder options (NEW) ===
if "non" in OUT_DIR:
    USE_PRETRAINED_ENCODER = False  # train from scratch
else:
    PRETRAINED_ENCODER_PATH = "results_seq/pretrained_patchtst_validation.pt"
ENCODER_LR              = 3e-4  # allow modest encoder fine-tuning
FREEZE_ENCODER          = False # fine-tune the encoder with the forecaster

# === Use Chronos-Bolt PEFT encoder instead of PatchTST (NEW) ===
if "patch" in OUT_DIR:
    USE_CB_ENCODER          = False
else:
    USE_CB_ENCODER          = True
# CB_ENCODER_CKPT         = "results_seq/pretrained_cb_encoder_peft_aug_H48.pt"
# CB_ENCODER_CKPT         = "results_seq/pretrained_cb_encoder_peft_aug_100best.pt"  # your PEFT checkpoint path
# CB_ENCODER_CKPT = "results_lora_variables-only/pretrained_cb_encoder_peft_independent_H24.pt"
# CB_ENCODER_CKPT         = "results_lora_new/pretrained_cb_encoder_peft_independent_H24.pt"   # best so far
CB_ENCODER_CKPT = "results_lora/new/pretrained_cb_encoder_peft_independent_H24.pt"
# === Day split for in-train validation (NEW) ===
TRAIN_LAST_DAY = 80   # use days 1..80 for training; 81..end for validation

# === Event-stratified validation settings (NEW) ===
EVENT_METHOD   = "mean"   # or "max"
STORM_TOP_Q    = 0.80     # top 20% days = stormy
CALM_BOTTOM_Q  = 0.50     # bottom 50% pool = calm
CALM_N         = 10       # sample 10 calm days
BEST_BY        = "overall"  # "overall" | "calm" | "storm"

# ========= NEW for zero/nonzero diagnostics =========
ZERO_EPS = 1e-8  # numerical tolerance for "zero" vs "nonzero"

# ========= NEW: MoE settings (minimal changes elsewhere) =========
USE_MOE        = True      # turn MoE readout on/off
MOE_NUM_EXPERTS= 4         # number of expert MLPs
MOE_TOP_K      = 2         # top-k routing (1 or 2 common)
MOE_HIDDEN     = 128       # hidden width inside each expert MLP
MOE_DROPOUT    = 0.10
W_MOE_AUX      = 1e-2      # weight for MoE load-balancing aux loss (KL to uniform)

# ========= NEW: Probabilistic head with quantiles =========
USE_QUANTILES   = True                       # if False -> classic point MSE
QUANTILES       = [0.1, 0.5, 0.9]            # any list of (0,1); 0.5 is median
PRED_QUANTILE   = 0.90                       # which quantile to turn into a point forecast for eval
# You can change to 0.5 (median) for less aggressive spikes.

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
        self.d_model = d_out  # for downstream code expecting .d_model

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

        hidden_dim = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if hidden_dim is None:
            raise RuntimeError("Cannot determine Chronos hidden size from config.")
        self.embed_proj = nn.Linear(self.in_channels, hidden_dim)
        self.post_proj  = nn.Linear(hidden_dim, self.d_model)

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
            # No spatial mixing; identity pass through with mixing layer
            return self.dropout(self.mix(x))

        B, T, N, C = x.shape
        # Build powers: [I, A, A^2, ..., A^K]
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
        # x: (B, T, N, F_in)
        h0 = self.in_proj(x)                 # (B, T, N, d_model)
        ht = self.tconv(h0)                  # temporal modeling
        A  = self.adaptA()                   # (N, N)
        hg = self.gdiff(ht, A)               # (B, T, N, d_model)
        h  = self.out_norm(h0 + hg)          # residual
        return h

# -------------------- NEW: MoE readout head (now supports arbitrary out_dim) --------------------
class MoEReadout(nn.Module):
    """
    Minimal top-k router + MLP experts.
    Inputs: h_last (B, N, D)
    Outputs: y (B, N, OUT_DIM), aux_loss (scalar), probs_mean (E,)
    """
    def __init__(self, d_in, out_dim, num_experts=4, hidden=128, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(max(1, min(top_k, self.num_experts)))
        self.router = nn.Linear(d_in, self.num_experts)  # per-node gating
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim)
            ) for _ in range(self.num_experts)
        ])

    def forward(self, h_last):
        # h_last: (B, N, D)
        B, N, D = h_last.shape
        logits = self.router(h_last)                 # (B, N, E)
        probs  = torch.softmax(logits, dim=-1)       # (B, N, E)

        # top-k routing (dense combine of selected experts)
        if self.top_k < self.num_experts:
            topv, topi = torch.topk(probs, k=self.top_k, dim=-1)   # (B, N, K)
            norm = torch.clamp(topv.sum(dim=-1, keepdim=True), min=1e-9)
            topw = topv / norm                                     # (B, N, K)
        else:
            # fall back to dense combine
            topi = torch.arange(self.num_experts, device=h_last.device).view(1,1,self.num_experts).expand(B,N,self.num_experts)
            topw = probs

        # expert outputs stacked: (B, N, OUT_DIM, E)
        expert_outs = torch.stack([exp(h_last) for exp in self.experts], dim=-1)

        # gather top-k expert outputs: (B,N,OUT_DIM,K)
        gather_idx = topi.unsqueeze(-2).expand(B, N, expert_outs.size(-2), self.top_k if self.top_k < self.num_experts else self.num_experts)
        selected = expert_outs.gather(dim=-1, index=gather_idx)

        # combine with weights
        y = (selected * topw.unsqueeze(-2)).sum(dim=-1)            # (B, N, OUT_DIM)

        # auxiliary load-balancing loss: KL(probs_mean || uniform)
        probs_mean = probs.mean(dim=(0,1))                         # (E,)
        eps = 1e-9
        aux = (probs_mean * (torch.log(probs_mean + eps) + math.log(self.num_experts))).sum()

        return y, aux, probs_mean

# -------------------- StemGNN Forecaster (PatchTST or CB encoder) --------------------
class STGNNForecaster(nn.Module):
    """
    Encoder -> stacked Stem blocks -> temporal pooling -> (MoE or Linear) readout.
    Input:  x (B, T, N, F_in)
    Output:
      - if USE_QUANTILES: (B, N, H, Q)
      - else:             (B, N, H)
    """
    def __init__(self, in_channels, horizon, num_nodes):
        super().__init__()
        self.num_nodes   = num_nodes
        self.horizon     = horizon
        self.in_channels = in_channels
        self.n_quantiles = len(QUANTILES) if USE_QUANTILES else 1
        out_dim = horizon * self.n_quantiles

        if USE_CB_ENCODER and os.path.exists(CB_ENCODER_CKPT):
            self.patch = ChronosBoltEncoderPEFT(in_channels=in_channels, ckpt_path=CB_ENCODER_CKPT)
            print("[Encoder] Using Chronos-Bolt PEFT encoder.")
        else:
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
            if 'PRETRAINED_ENCODER_PATH' in globals() and PRETRAINED_ENCODER_PATH is not None and os.path.exists(PRETRAINED_ENCODER_PATH):
                try:
                    ckpt = torch.load(PRETRAINED_ENCODER_PATH, map_location="cpu")
                    state = ckpt.get("state_dict", ckpt)
                    self.patch.load_state_dict(state, strict=False)
                    print(f"[Encoder] Loaded pretrained PatchTST from: {PRETRAINED_ENCODER_PATH}")
                except Exception as e:
                    print(f"[Encoder] Warning: failed to load pretrained PatchTST: {e}")

        if FREEZE_ENCODER:
            for p in self.patch.parameters():
                p.requires_grad = False
            print("[Encoder] Frozen (no fine-tuning).")

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

        # ===== choose readout =====
        if USE_MOE:
            self.readout_moe = MoEReadout(
                d_in=STEM_D_MODEL, out_dim=out_dim,
                num_experts=MOE_NUM_EXPERTS, hidden=MOE_HIDDEN,
                top_k=MOE_TOP_K, dropout=MOE_DROPOUT
            )
            self.readout = None
            print(f"[MoE] Enabled: E={MOE_NUM_EXPERTS}, top_k={MOE_TOP_K}, hidden={MOE_HIDDEN}, out_dim={out_dim}")
        else:
            self.readout = nn.Linear(STEM_D_MODEL, out_dim)
            self.readout_moe = None

        # buffers for aux stats
        self._last_aux_loss = None
        self._last_gate_mean = None

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        assert N == self.num_nodes and F == self.in_channels, \
            f"StemGNN input shape mismatch: got N={N},F={F}, expected N={self.num_nodes},F={self.in_channels}"

        h = self.patch(x)
        for blk in self.blocks:
            h = blk(h)

        last = h[:, -1]  # (B, N, D)

        if self.readout_moe is not None:
            y_flat, aux, probs_mean = self.readout_moe(last)  # (B,N,H*Q)
            self._last_aux_loss = aux
            self._last_gate_mean = probs_mean.detach()
        else:
            y_flat = self.readout(last)                       # (B,N,H*Q or H)
            self._last_aux_loss = None
            self._last_gate_mean = None

        if USE_QUANTILES:
            y = y_flat.view(B, N, self.horizon, self.n_quantiles)  # (B,N,H,Q)
            return y
        else:
            return y_flat  # (B,N,H)

    # convenience getters
    def moe_aux_loss(self):
        return self._last_aux_loss

    def moe_gate_usage(self):
        return self._last_gate_mean

# ---------------- helpers for day split & windows (NEW) ----------------
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
      Y_tr: (num_windows_total, 1, horizon)
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

        # --- per-county inference with N=1 ---
        for i, county in enumerate(locs):
            Xin_i = X_all[start_ctx:end_idx+1, i, :].reshape(1, SEQ_LEN, 1, X_all.shape[2])  # (1, SEQ_LEN, 1, Din)
            pred_sc = infer(model, Xin_i)[0, 0]  # (H,)

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

# ========= NEW: zero/nonzero diagnostic ratios =========
# ========= REPLACE your _zero_nonzero_ratios with this version =========
def _zero_nonzero_ratios(df_truth_wide: pd.DataFrame, df_pred_long: pd.DataFrame, eps: float = ZERO_EPS):
    """
    Returns (zero->nonzero_ratio, nonzero->zero_ratio)
      zero->nonzero: fraction of true==0 cells where pred>eps  (false positives)
      nonzero->zero: fraction of true>eps cells where pred<=eps (false negatives)
    Operates on the intersection of timestamps & locations present in both.
    Robust to duplicate (timestamp, location) rows in df_pred_long.
    """
    if df_truth_wide is None or df_truth_wide.empty or df_pred_long is None or df_pred_long.empty:
        return float("nan"), float("nan")

    # ---- Coerce & clean truth (wide) ----
    df_t = df_truth_wide.copy()
    df_t["timestamp"] = pd.to_datetime(df_t["timestamp"], errors="coerce").astype("datetime64[ns]")
    df_t = df_t.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    tw = df_t.set_index("timestamp").sort_index()

    # ---- Coerce & clean preds (long) ----
    df_p = df_pred_long.copy()
    df_p["timestamp"] = pd.to_datetime(df_p["timestamp"], errors="coerce").astype("datetime64[ns]")
    df_p["location"]  = df_p["location"].astype(str)
    df_p = df_p.dropna(subset=["timestamp", "location"])

    # Collapse duplicate (timestamp, location) pairs (mean)
    df_p = df_p.groupby(["timestamp", "location"], as_index=False, sort=False)["pred"].mean()

    # Pivot with aggregation safety
    pw = df_p.pivot_table(index="timestamp", columns="location", values="pred", aggfunc="mean")
    pw = pw.sort_index()

    # ---- Align intersection (timestamps / locations) ----
    common_ts = tw.index.intersection(pw.index)
    if len(common_ts) == 0:
        return float("nan"), float("nan")
    tw = tw.loc[common_ts]
    pw = pw.loc[common_ts]

    # Align locations
    loc_truth = list(tw.columns)
    loc_pred  = list(pw.columns)
    locs = sorted(set(loc_truth).intersection(loc_pred))
    if len(locs) == 0:
        return float("nan"), float("nan")

    y_true = tw[locs].to_numpy()
    y_pred = pw[locs].to_numpy()

    # valid mask
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan"), float("nan")

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    true_zero = (np.abs(y_true) <= eps)
    pred_zero = (np.abs(y_pred) <= eps)

    # zero -> nonzero (false positives among truly zero)
    denom_zero = np.sum(true_zero)
    z_to_nz = np.sum(true_zero & (~pred_zero)) / denom_zero if denom_zero > 0 else float("nan")

    # nonzero -> zero (false negatives among truly nonzero)
    denom_nz = np.sum(~true_zero)
    nz_to_z = np.sum((~true_zero) & pred_zero) / denom_nz if denom_nz > 0 else float("nan")

    return float(z_to_nz), float(nz_to_z)

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

def make_val_callback_rolling(ds_train, scalers, horizon, aux, label,
                              train_last_day=80,
                              ts_future_test=None, df_truth_test=None, zero_rmse_test=float("nan")):
    """
    Evaluate each epoch on rolling day-ahead validation (days 81..end) **and** on the test set.
    Prints:
      - Rolling RMSE and all-zero baseline
      - Calm-heavy & Storm-heavy RMSEs + all-zero baselines
      - Zero/Nonzero ratios (model vs all-zero baseline)
      - Testing RMSE (with all-zero baseline) and test ratios
    Saves best checkpoint by metric selected via BEST_BY.
    """
    best = {"rmse": float("inf"), "epoch": -1}
    df_truth_val = truth_wide_for_val(ds_train, aux, horizon, train_last_day)

    def _callback(model, epoch):
        df_pred_val = predict_val_roll(ds_train, model, scalers, horizon, aux, train_last_day)

        # event-stratified metrics (with all-zero baselines)
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

        # ---- NEW: zero/nonzero ratios (overall) ----
        zpnz_model, nzpz_model = _zero_nonzero_ratios(df_truth_val, df_pred_val, eps=ZERO_EPS)
        # baseline: build zero preds matching truth timestamps/locations
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

        # Print nicely
        if np.isfinite(rmse_overall):
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE={rmse_overall:.4f} (all-zero={zero_overall:.4f})")
        else:
            print(f"[Val {label}] Epoch {epoch}: Rolling RMSE=nan (all-zero=nan)")

        # NEW: print ratios
        print(f"[Val {label}] Epoch {epoch}: Zero→Nonzero ratio={zpnz_model if np.isfinite(zpnz_model) else float('nan'):.4f} "
              f"(all-zero={zpnz_zero if np.isfinite(zpnz_zero) else float('nan'):.4f}) | "
              f"Nonzero→Zero ratio={nzpz_model if np.isfinite(nzpz_model) else float('nan'):.4f} "
              f"(all-zero={nzpz_zero if np.isfinite(nzpz_zero) else float('nan'):.4f})")

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
            "overall": rmse_overall,
            "calm": rmse_calm,
            "storm": rmse_storm
        }
        sel = metric_map.get(str(BEST_BY).lower(), rmse_overall)
        sel_val = sel if np.isfinite(sel) else rmse_overall  # fallback to overall if selected is nan

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

            # Save predictions, ground truths, and per-day plots
            ts_map = pd.DataFrame({
                "timestamp": pd.to_datetime(aux["timestamps"]),
                "day": aux["days"]
            })
            df_pred_day = df_pred_val.merge(ts_map, on="timestamp", how="left")
            df_truth_day = df_truth_val.merge(ts_map, on="timestamp", how="left")

            unique_days = sorted(df_pred_day["day"].dropna().unique())
            for d in unique_days:
                pred_d = df_pred_day[df_pred_day["day"] == d].drop(columns="day")
                truth_d = df_truth_day[df_truth_day["day"] == d].drop(columns="day")
                pred_path = os.path.join(OUT_DIR, f"val_pred_epoch{epoch}_day{int(d)}_{label}.csv")
                truth_path = os.path.join(OUT_DIR, f"val_truth_epoch{epoch}_day{int(d)}_{label}.csv")
                pred_d.to_csv(pred_path, index=False)
                truth_d.to_csv(truth_path, index=False)
                print(f"[Val {label}] Saved day {int(d)} prediction/truth data: {pred_path}, {truth_path}")

            if unique_days:
                fig, axes = plt.subplots(len(unique_days), 1, figsize=(8, 3 * len(unique_days)), sharex=False)
                if len(unique_days) == 1:
                    axes = [axes]
                for ax, d in zip(axes, unique_days):
                    truth_d = df_truth_day[df_truth_day["day"] == d].set_index("timestamp").drop(columns="day")
                    truth_mean = truth_d.mean(axis=1).reset_index(name="truth")
                    pred_mean = (
                        df_pred_day[df_pred_day["day"] == d]
                        .groupby("timestamp")["pred"].mean().reset_index()
                    )
                    merged = pd.merge(truth_mean, pred_mean, on="timestamp", how="inner").sort_values("timestamp")
                    ax.plot(merged["timestamp"], merged["truth"], label="Truth")
                    ax.plot(merged["timestamp"], merged["pred"], label="Prediction")
                    ax.set_title(f"Day {int(d)}")
                    ax.set_xlabel("timestamp")
                    ax.set_ylabel("outage")
                    ax.legend()
                fig_path = os.path.join(OUT_DIR, f"val_curve_epoch{epoch}_{label}.png")
                fig.tight_layout()
                fig.savefig(fig_path)
                plt.close(fig)
                print(f"[Val {label}] Saved prediction vs truth plot: {fig_path}")

        # ---- Test set evaluation at each validation step ----
        if ts_future_test is not None and df_truth_test is not None and len(ts_future_test) > 0:
            df_pred_test = predict_per_county(ds_train, model, scalers, horizon, ts_future_test)
            rmse_test = per_county_rmse_mean(df_truth_test, df_pred_test)

            # NEW: test ratios vs all-zero baseline
            zpnz_t, nzpz_t = _zero_nonzero_ratios(df_truth_test, df_pred_test, eps=ZERO_EPS)
            locs_t = [c for c in df_truth_test.columns if c != "timestamp"]
            df_zero_long_t = pd.DataFrame({
                "timestamp": np.repeat(df_truth_test["timestamp"].values, len(locs_t)),
                "location": np.tile(locs_t, len(df_truth_test)),
                "pred": 0.0
            })
            zpnz_t0, nzpz_t0 = _zero_nonzero_ratios(df_truth_test, df_zero_long_t, eps=ZERO_EPS)

            print(f"[Val {label}] Epoch {epoch}: Testing RMSE={rmse_test:.4f} (all-zero RMSE={zero_rmse_test:.4f})")
            print(f"[Val {label}] Epoch {epoch}: Test Zero→Nonzero ratio={zpnz_t if np.isfinite(zpnz_t) else float('nan'):.4f} "
                  f"(all-zero={zpnz_t0 if np.isfinite(zpnz_t0) else float('nan'):.4f}) | "
                  f"Nonzero→Zero ratio={nzpz_t if np.isfinite(nzpz_t) else float('nan'):.4f} "
                  f"(all-zero={nzpz_t0 if np.isfinite(nzpz_t0) else float('nan'):.4f})")

    return _callback

# ---------------- NEW: pinball loss for quantile regression ----------------
def _pinball_loss(yq, y_true, q_list):
    """
    yq:     (B, N, H, Q)  predicted quantiles
    y_true: (B, N, H)     target
    q_list: list/tuple of quantiles in (0,1)
    Returns scalar loss averaged over all dims.
    """
    qs = torch.tensor(q_list, device=yq.device, dtype=yq.dtype).view(1,1,1,-1)  # (1,1,1,Q)
    diff = y_true.unsqueeze(-1) - yq                                           # (B,N,H,Q)
    loss = torch.maximum(qs * diff, (qs - 1.0) * diff)                         # (B,N,H,Q)
    return loss.mean()

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

    enc_params = list(model.patch.parameters())
    enc_ids = set(id(p) for p in enc_params)
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]

    if FREEZE_ENCODER:
        opt = torch.optim.Adam(other_params, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        opt = torch.optim.Adam(
            [
                {"params": enc_params,   "lr": ENCODER_LR},
                {"params": other_params, "lr": LR}
            ],
            weight_decay=WEIGHT_DECAY
        )

    loss_fn_point = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)  # xb: (B, T, N, F), yb: (B, N, H)

            if TEMPORAL_DROP_PROB > 0.0:
                with torch.no_grad():
                    drop_mask = (torch.rand(xb.size(0), xb.size(1), 1, 1, device=xb.device) < TEMPORAL_DROP_PROB).float()
                    xb = xb * (1.0 - drop_mask)

            opt.zero_grad()
            pred = model(xb)                       # (B, N, H) or (B, N, H, Q)

            if USE_QUANTILES:
                loss = _pinball_loss(pred, yb, QUANTILES)
            else:
                loss = loss_fn_point(pred, yb)

            # ===== add MoE aux loss if present =====
            aux = model.moe_aux_loss()
            if aux is not None:
                loss = loss + W_MOE_AUX * aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        elapsed = time.time() - start_time
        lr_str = ", ".join(f"{pg['lr']:.2e}" for pg in opt.param_groups)

        # (optional) quick router usage printout
        gate_mean = model.moe_gate_usage()
        if gate_mean is not None:
            gm = gate_mean.detach().cpu().numpy()
            gm_str = " | ".join([f"E{i}:{gm[i]:.2f}" for i in range(len(gm))])
            print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s | LR: {lr_str} | MoE usage={gm_str}")
        else:
            print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s | LR: {lr_str}")

        if eval_callback is not None:
            eval_callback(model, epoch + 1)

    return model

@torch.no_grad()
def infer(model, Xin):
    """
    Xin: (1, SEQ_LEN, N, F)
    Returns: (1, N, H) point forecast by selecting a quantile (if enabled).
    """
    model.eval()
    out = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE))
    if USE_QUANTILES:
        # (1, N, H, Q) -> select chosen quantile along last axis
        qs = np.array(QUANTILES, dtype=float)
        # pick index of quantile closest to PRED_QUANTILE
        idx = int(np.argmin(np.abs(qs - float(PRED_QUANTILE))))
        out = out[..., idx]  # (1, N, H)
    return out.cpu().numpy()

# ---------------- pipeline helpers (train windows only) ----------------
def prepare_training(ds_train, horizon):
    X, Y, Din, scalers, aux = prepare_training_split(ds_train, horizon, TRAIN_LAST_DAY)
    return X, Y, Din, scalers, aux

# ---------------- test-time prediction (updated for N=1 inference) ----------------
def predict_per_county(ds_train, model, scalers, horizon, ts_future):
    """
    Predict next 'horizon' for ALL counties, running the model per-county (N=1).
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
            pred_sc = infer(model, Xin_i)[0, 0]                         # (H,)

        pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))
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

            pred_path = os.path.join(OUT_DIR, f"val_pred_epoch{epoch}_{label}.csv")
            truth_path = os.path.join(OUT_DIR, f"val_truth_epoch{epoch}_{label}.csv")
            df_pred.to_csv(pred_path, index=False)
            df_truth.to_csv(truth_path, index=False)

            truth_mean = df_truth.set_index("timestamp").mean(axis=1).reset_index(name="truth")
            pred_mean = df_pred.groupby("timestamp")["pred"].mean().reset_index()
            merged = pd.merge(truth_mean, pred_mean, on="timestamp", how="inner").sort_values("timestamp")
            plt.figure()
            plt.plot(merged["timestamp"], merged["truth"], label="Truth")
            plt.plot(merged["timestamp"], merged["pred"], label="Prediction")
            plt.legend()
            plt.xlabel("timestamp")
            plt.ylabel("outage")
            plt.title(f"Validation Prediction vs Truth ({label}) epoch {epoch}")
            fig_path = os.path.join(OUT_DIR, f"val_curve_epoch{epoch}_{label}.png")
            plt.savefig(fig_path)
            plt.close()
            print(f"[Eval {label}] Saved prediction/truth data: {pred_path}, {truth_path}")
            print(f"[Eval {label}] Saved prediction vs truth plot: {fig_path}")

    return _callback

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
    X24, Y24, Din, scalers24, aux24 = prepare_training(ds_train, horizon=H24)
    n_nodes = 1  # <<< single-node per sample
    val_cb24 = make_val_callback_rolling(
        ds_train, scalers24, H24, aux24, label="24h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts24, df_truth_test=df_t24, zero_rmse_test=zero_rmse_24
    )
    mdl24 = train_model(X24, Y24, Din, H24, n_nodes, eval_callback=val_cb24)
    # mdl24 = None

    # ==== 48h ====
    X48, Y48, _, scalers48, aux48 = prepare_training(ds_train, horizon=H48)
    val_cb48 = make_val_callback_rolling(
        ds_train, scalers48, H48, aux48, label="48h", train_last_day=TRAIN_LAST_DAY,
        ts_future_test=ts48, df_truth_test=df_t48, zero_rmse_test=zero_rmse_48
    )
    mdl48 = train_model(X48, Y48, Din, H48, n_nodes, eval_callback=val_cb48)

    # ----- Test evaluation using best models, then write CSVs -----
    # 24h
    if mdl24 is not None:
        best24_path = os.path.join(OUT_DIR, "best_enc-dec_24h.pt")
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
        df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24)
        test_rmse_24 = per_county_rmse_mean(df_t24, df24)
        print(f"[Test 24h] Testing RMSE={test_rmse_24:.4f} (all-zero RMSE={zero_rmse_24:.4f})")

        # NEW: final test ratios for 24h
        zpnz_24, nzpz_24 = _zero_nonzero_ratios(df_t24, df24, eps=ZERO_EPS)
        locs24 = [c for c in df_t24.columns if c != "timestamp"]
        df_zero_long_24 = pd.DataFrame({
            "timestamp": np.repeat(df_t24["timestamp"].values, len(locs24)),
            "location": np.tile(locs24, len(df_t24)),
            "pred": 0.0
        })
        zpnz_24_0, nzpz_24_0 = _zero_nonzero_ratios(df_t24, df_zero_long_24, eps=ZERO_EPS)
        print(f"[Test 24h] Zero→Nonzero ratio={zpnz_24 if np.isfinite(zpnz_24) else float('nan'):.4f} "
              f"(all-zero={zpnz_24_0 if np.isfinite(zpnz_24_0) else float('nan'):.4f}) | "
              f"Nonzero→Zero ratio={nzpz_24 if np.isfinite(nzpz_24) else float('nan'):.4f} "
              f"(all-zero={nzpz_24_0 if np.isfinite(nzpz_24_0) else float('nan'):.4f})")
    else:
        df24 = None

    # 48h
    if mdl48 is not None:
        best48_path = os.path.join(OUT_DIR, "best_enc-dec_48h.pt")
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
        df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48)
        test_rmse_48 = per_county_rmse_mean(df_t48, df48)
        print(f"[Test 48h] Testing RMSE={test_rmse_48:.4f} (all-zero RMSE={zero_rmse_48:.4f})")

        # NEW: final test ratios for 48h
        zpnz_48, nzpz_48 = _zero_nonzero_ratios(df_t48, df48, eps=ZERO_EPS)
        locs48 = [c for c in df_t48.columns if c != "timestamp"]
        df_zero_long_48 = pd.DataFrame({
            "timestamp": np.repeat(df_t48["timestamp"].values, len(locs48)),
            "location": np.tile(locs48, len(df_t48)),
            "pred": 0.0
        })
        zpnz_48_0, nzpz_48_0 = _zero_nonzero_ratios(df_t48, df_zero_long_48, eps=ZERO_EPS)
        print(f"[Test 48h] Zero→Nonzero ratio={zpnz_48 if np.isfinite(zpnz_48) else float('nan'):.4f} "
              f"(all-zero={zpnz_48_0 if np.isfinite(zpnz_48_0) else float('nan'):.4f}) | "
              f"Nonzero→Zero ratio={nzpz_48 if np.isfinite(nzpz_48) else float('nan'):.4f} "
              f"(all-zero={nzpz_48_0 if np.isfinite(nzpz_48_0) else float('nan'):.4f})")
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

    out24 = os.path.join(OUT_DIR, "stemgnn_pred_24h.csv")
    out48 = os.path.join(OUT_DIR, "stemgnn_pred_48h.csv")
    df24.to_csv(out24, index=False)
    df48.to_csv(out48, index=False)

    print("Saved predictions:")
    print(f"  {out24}")
    print(f"  {out48}")

    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
