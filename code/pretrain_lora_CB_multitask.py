import os
import time
from math import ceil
from tqdm import tqdm
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =========================
# Config
# =========================
SPLIT_DIR   = "data/"
OUT_DIR     = "results_seq/"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Windows / data
SEQ_LEN     = 24     # context length T (match downstream)
H_PRE       = 24     # pretraining forecast horizon H
BATCH_SIZE  = 32

# Training
EPOCHS          = 20
LR              = 1e-4
WEIGHT_DECAY    = 1e-3
GRAD_CLIP_NORM  = 5.0

# ---------- Masking for SSL ----------
MASK_MODE               = "span"   # "random" or "span"
MASK_RATIO              = 0.30     # fraction of timesteps masked per window
SPAN_LEN                = 4        # used if MASK_MODE=="span": contiguous length per span

# For contrastive views (lighter masking by default)
VIEW_MASK_MODE          = "span"   # "random" or "span"
VIEW_MASK_RATIO         = 0.15
VIEW_SPAN_LEN           = 3

# ---------- Contrastive (InfoNCE) ----------
USE_CONTRASTIVE         = True
W_CONTRAST              = 1.0      # weight of contrastive loss
TAU                     = 0.2      # temperature for InfoNCE
ADD_JITTER_STD          = 0.01     # small Gaussian jitter std for views
ADD_DROPOUT_PROB        = 0.0      # channel dropout prob for views
# If you want contrastive gradients to update the encoder, set to True
CONTRAST_UPDATES_ENCODER = False

# ---------- Frequency-domain augmentations (TimesNet-style) ----------
USE_FREQ_AUG            = True
FREQ_AUG_PROB           = 0.8      # chance to apply frequency aug per view
FREQ_AMP_STD            = 0.10     # multiplicative magnitude noise ~ N(1, std)
FREQ_PHASE_STD          = 0.20     # phase jitter (radians) ~ N(0, std)
FREQ_DROPOUT_FRAC       = 0.10     # fraction of frequency bins to zero
FREQ_DROPOUT_PROB       = 0.5      # prob of applying spectral dropout
FREQ_BAND_PROB          = 0.5      # prob of band-stop (else band-keep) op
FREQ_BAND_WIDTH_RATIO   = (0.05, 0.25)  # width as fraction of Nyquist [min, max]
FREQ_SHELF_PROB         = 0.3      # prob of applying high/low shelf attenuation
FREQ_SHELF_STRENGTH     = (0.3, 0.8)    # attenuation factor range

# ---------- Multitask loss weights ----------
W_SSL       = 1.0   # masked reconstruction loss weight
W_FORECAST  = 1.0   # next-H forecasting loss weight

# ---------- CB + PEFT (LoRA) ----------
CB_MODEL_NAME   = "/home2/caojia/weather_prediction/code/chronos-bolt-small"  # or a local path
D_OUT           = 64                          # encoder output dim (match downstream STEM_D_MODEL if desired)
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

# Save path (wrapper state_dict)
SAVE_PATH = os.path.join(OUT_DIR, "pretrained_cb_encoder_peft_multitask_contrast_freq.pt")
# Optional: export LoRA adapter/backbone in HF format
SAVE_ADAPTER_DIR = os.path.join(OUT_DIR, "cb_peft_adapter_multitask")


# =========================
# Utils
# =========================
def open_ds(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path, engine="netcdf4")

def zfit(arr):
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def zapply(arr, mu, sd):
    return (arr - mu) / sd


# =========================
# Data prep: windows for SSL + forecasting
# =========================
def prepare_multitask_windows(ds_train, seq_len=24, horizon=24):
    """
    Returns:
      X     : (nW, T, N, Din)                      inputs (scaled)
      Y_fut : (nW, H, N, Din)                      future targets for all channels
      Din   : 1 (outage) + F (weather)
      scalers: dict
    """
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    T, N, F = w.shape
    Din = 1 + F

    # scale
    y_mu, y_sd = zfit(y.reshape(-1, 1))
    w_mu, w_sd = zfit(w.reshape(-1, F))
    y_sc = zapply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, N)        # (T, N)
    w_sc = zapply(w.reshape(-1, F), w_mu, w_sd).reshape(T, N, F)     # (T, N, F)

    # windows
    nW = T - seq_len - horizon + 1
    if nW <= 0:
        raise RuntimeError("Not enough history for the given SEQ_LEN and horizon.")

    X_all = np.concatenate([y_sc[..., None], w_sc], axis=2)          # (T, N, Din)
    X     = np.stack([X_all[i:i+seq_len] for i in range(nW)], axis=0)          # (nW, T, N, Din)
    Y_fut = np.stack([X_all[i+seq_len:i+seq_len+horizon] for i in range(nW)], axis=0)  # (nW, H, N, Din)

    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    return X, Y_fut, Din, scalers


# =========================
# Chronos-Bolt PEFT encoder
# =========================
try:
    from transformers import AutoModel, AutoConfig
    HAVE_TRANSFORMERS = True
except Exception as e:
    HAVE_TRANSFORMERS = False
    _transformers_err = str(e)

try:
    from peft import LoraConfig, get_peft_model
    HAVE_PEFT = True
except Exception as e:
    HAVE_PEFT = False
    _peft_err = str(e)


def _discover_lora_targets(model: nn.Module):
    """
    Inspect Linear submodules and pick target substrings that actually exist.
    Covers T5-like & common attention names, with safe fallbacks.
    """
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    candidates = [
        # LLaMA-ish
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Falcon/NeoX-ish
        "query_key_value", "out_proj", "in_proj",
        # Diffusers/misc
        "to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out",
        # T5-ish leaves
        "q", "k", "v", "o", "wi", "wo", "wi_0", "wi_1",
        # Generic/fallbacks
        "Wqkv", "Wo", "Wq", "Wk", "Wv",
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
    Wrap a Chronos/Chronos-Bolt backbone with LoRA (PEFT) and simple
    input/output adapters to produce (B, T, N, D_OUT) embeddings.
    """
    def __init__(self, in_channels, cb_model_name=CB_MODEL_NAME, d_out=D_OUT,
                 lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT):
        super().__init__()
        if not HAVE_TRANSFORMERS:
            raise ImportError(
                f"transformers not available: {_transformers_err}\n"
                "Install: pip install transformers accelerate safetensors"
            )
        if not HAVE_PEFT:
            raise ImportError(
                f"peft not available: {_peft_err}\n"
                "Install: pip install peft"
            )

        self.in_channels = in_channels
        self.d_out = d_out

        self.config = AutoConfig.from_pretrained(cb_model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            cb_model_name,
            trust_remote_code=True,
            output_hidden_states=True
        )

        targets = _discover_lora_targets(self.backbone)
        print(f"[PEFT] LoRA targets: {targets}")

        lcfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        try:
            self.backbone = get_peft_model(self.backbone, lcfg)
        except ValueError as e:
            print(f"[PEFT] Warning: {e} -> fallback target_modules=['linear']")
            lcfg.target_modules = ["linear"]
            self.backbone = get_peft_model(self.backbone, lcfg)

        hidden_dim = getattr(self.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.config, "d_model", None)
        if hidden_dim is None:
            raise RuntimeError("Cannot determine Chronos hidden size from config.")

        # Adapt multivariate input into backbone hidden dim (per step)
        self.embed_proj = nn.Linear(in_channels, hidden_dim)
        # Map backbone hidden to D_OUT
        self.post_proj  = nn.Linear(hidden_dim, d_out)

    def forward(self, x):
        """
        x: (B, T, N, F) -> (B, T, N, D_OUT)
        Use inputs_embeds into encoder (works for T5-like models).
        """
        B, T, N, F = x.shape
        assert F == self.in_channels, f"CB encoder expected F={self.in_channels}, got {F}"

        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)   # (B*N, T, F)
        emb  = self.embed_proj(x_bn)                        # (B*N, T, H)
        attn_mask = torch.ones(B * N, T, dtype=torch.long, device=emb.device)

        if hasattr(self.backbone, "encoder") and callable(self.backbone.encoder):
            out = self.backbone.encoder(
                inputs_embeds=emb,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        else:
            out = self.backbone(
                inputs_embeds=emb,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            h_bn = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            h_bn = out.hidden_states[-1]
        else:
            if isinstance(out, (tuple, list)) and len(out) > 0:
                h_bn = out[0]
            else:
                raise RuntimeError("Backbone forward did not return hidden states.")

        # Ensure time alignment
        if h_bn.size(1) > T:
            h_bn = h_bn[:, :T, :]
        elif h_bn.size(1) < T:
            pad_len = T - h_bn.size(1)
            pad = torch.zeros(h_bn.size(0), pad_len, h_bn.size(2), device=h_bn.device, dtype=h_bn.dtype)
            h_bn = torch.cat([h_bn, pad], dim=1)

        h_bn = self.post_proj(h_bn)                # (B*N, T, D_OUT)
        h = h_bn.view(B, N, T, self.d_out).permute(0, 2, 1, 3)  # (B, T, N, D_OUT)
        return h

    def print_trainable(self):
        try:
            self.backbone.print_trainable_parameters()
        except Exception:
            total, trainable = 0, 0
            for p in self.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            print(f"[PEFT] Trainable params {trainable}/{total} ({100.0*trainable/total:.2f}%)")


# =========================
# Multi-task model using CB encoder
# =========================
class MultiTaskCBEncoder(nn.Module):
    """
    Encoder + heads:
      - recon_head: reconstruct inputs at masked timesteps (SSL)
      - forecast_head: predict next H steps of full multivariate inputs
      - proj_head: projection for InfoNCE (contrastive)
    """
    def __init__(self, in_channels, horizon):
        super().__init__()
        self.horizon = horizon
        self.encoder = ChronosBoltEncoderPEFT(in_channels=in_channels, d_out=D_OUT)

        # SSL reconstruction head: (B, T, N, D) -> (B, T, N, F)
        self.recon_head = nn.Linear(D_OUT, in_channels)

        # Forecast head: pool over time per node, then MLP -> (H * F)
        self.forecast_head = nn.Sequential(
            nn.Linear(D_OUT, 2 * D_OUT),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * D_OUT, horizon * in_channels)
        )

        # Projection head for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(D_OUT, D_OUT),
            nn.GELU(),
            nn.Linear(D_OUT, D_OUT)
        )

    def encode_steps(self, x):
        """Return per-step embeddings (B, T, N, D_OUT)."""
        return self.encoder(x)

    def forward(self, x):
        # x: (B, T, N, F)
        h = self.encoder(x)                    # (B, T, N, D_OUT)

        # Reconstruction at each past step
        recon = self.recon_head(h)             # (B, T, N, F)

        # Forecast next H steps for all channels
        pooled = h.mean(dim=1)                 # (B, N, D_OUT)
        fut = self.forecast_head(pooled)       # (B, N, H*F)
        B, N, _ = fut.shape
        fut = fut.view(B, N, self.horizon, -1).permute(0, 2, 1, 3)  # (B, H, N, F)

        return recon, fut

    def pooled_projection(self, x):
        """
        For contrastive: encode -> pool across T and N -> projection -> L2 normalize.
        If CONTRAST_UPDATES_ENCODER is False, detach to avoid encoder grads here.
        """
        h = self.encoder(x) if CONTRAST_UPDATES_ENCODER else self.encoder(x).detach()
        pooled = h.mean(dim=1).mean(dim=1)       # (B, D_OUT)
        z = self.proj_head(pooled)               # (B, D_OUT)
        z = F.normalize(z, dim=-1)
        return z


# =========================
# Losses
# =========================
def masked_recon_loss(pred, target, mask):
    """
    pred/target: (B, T, N, F)
    mask:       (B, T, 1, 1) 1 where masked (reconstruct), 0 otherwise
    """
    m = mask.expand_as(pred)
    denom = m.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * m).sum() / denom

def forecast_loss(pred, target):
    """
    pred:   (B, H, N, F)
    target: (B, H, N, F)
    """
    return torch.mean((pred - target) ** 2)

def info_nce_loss(z1, z2, tau=0.2):
    """
    NT-Xent across batch. z1,z2: (B, D), L2-normalized.
    """
    B = z1.size(0)
    if B == 1:
        return torch.tensor(0.0, device=z1.device)
    sim = (z1 @ z2.t()) / tau           # (B, B)
    labels = torch.arange(B, device=z1.device)
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_12 + loss_21)


# =========================
# Masking & Augmentations
# =========================
def make_temporal_mask(batch_size, T, mode="random", ratio=0.3, span_len=4, device="cpu"):
    """
    Returns mask of shape (B, T, 1, 1) with 1.0 at masked timesteps.
    - random: IID Bernoulli per step with prob=ratio.
    - span:   choose K spans of length span_len covering approx ratio of T.
    """
    if mode == "random":
        m = (torch.rand(batch_size, T, 1, 1, device=device) < ratio).float()
        return m

    # span mode
    K = max(1, ceil((ratio * T) / max(1, span_len)))
    m = torch.zeros(batch_size, T, 1, 1, device=device)
    for b in range(batch_size):
        for _ in range(K):
            if T <= span_len:
                start = 0
            else:
                start = torch.randint(0, T - span_len + 1, (1,), device=device).item()
            m[b, start:start+span_len, :, :] = 1.0
    return m.clamp(max=1.0)

def apply_mask(x, mask):
    """Zero-out masked timesteps. x: (B,T,N,F); mask: (B,T,1,1)."""
    return x * (1.0 - mask)

def add_jitter(x, std=0.01):
    """Add small Gaussian noise."""
    if std <= 0.0:
        return x
    noise = torch.randn_like(x) * std
    return x + noise

def dropout_features(x, p=0.0):
    """Randomly drop features per timestep (channel dropout)."""
    if p <= 0.0:
        return x
    B, T, N, F = x.shape
    mask = (torch.rand(B, 1, 1, F, device=x.device) < p).float()
    return x * (1.0 - mask)

# ====== Frequency-domain augmentations (TimesNet-style) ======
@torch.no_grad()
def _variance_match(x_aug, x_ref, eps=1e-6):
    """Rescale x_aug to match the per-sample std of x_ref."""
    s_ref = x_ref.flatten(1).std(dim=1, keepdim=True).clamp_min(eps)          # (B,1)
    s_aug = x_aug.flatten(1).std(dim=1, keepdim=True).clamp_min(eps)          # (B,1)
    scale = (s_ref / s_aug).view(-1, 1, 1, 1)
    return x_aug * scale

def _random_band_mask(B, Lf, device, keep=False):
    """Create (B, Lf, 1, 1) band mask. keep=False: band-stop; keep=True: band-keep."""
    centers = torch.rand(B, device=device) * (Lf - 1)
    widths  = torch.empty(B, device=device).uniform_(
        int(FREQ_BAND_WIDTH_RATIO[0] * Lf),
        max(1.0, FREQ_BAND_WIDTH_RATIO[1] * Lf)
    )
    idx = torch.arange(Lf, device=device).view(1, Lf)
    mask = torch.ones(B, Lf, 1, 1, device=device)
    for b in range(B):
        c = centers[b]
        w = widths[b]
        lo = (c - w / 2.0).clamp(0, Lf - 1)
        hi = (c + w / 2.0).clamp(0, Lf - 1)
        band = ((idx >= lo) & (idx <= hi)).float().view(1, Lf, 1, 1)
        mask[b:b+1] = band if keep else (1.0 - band)
    return mask

def _shelf_curve(Lf, high=True, strength=0.5, device="cpu"):
    """High/low shelf attenuation curve over frequencies. (1, Lf, 1, 1)"""
    f = torch.linspace(0.0, 1.0, steps=Lf, device=device).view(1, Lf, 1, 1)
    if high:
        curve = 1.0 - strength * f.pow(1.0)          # attenuate high freq
    else:
        curve = 1.0 - strength * (1.0 - f).pow(1.0)  # attenuate low freq
    return curve.clamp_min(0.0)

def freq_augment(x, prob=FREQ_AUG_PROB):
    """
    Mix of frequency-domain augs on x (B,T,N,F) with probability `prob`:
      - magnitude scaling in freq domain
      - phase jitter
      - band stop/keep
      - spectral dropout
      - shelving (high/low attenuation)
    Then normalize to match original variance.
    """
    if (not USE_FREQ_AUG) or (torch.rand(1).item() > prob):
        return x

    B, T, N, Fdim = x.shape
    Xf = torch.fft.rfft(x, dim=1)  # (B, Lf, N, F), complex
    Lf = Xf.size(1)

    # amplitude scaling
    if FREQ_AMP_STD > 0.0:
        amp = torch.randn(B, Lf, 1, 1, device=x.device, dtype=Xf.real.dtype) * FREQ_AMP_STD + 1.0
        Xf = Xf * amp

    # phase jitter
    if FREQ_PHASE_STD > 0.0:
        phi = torch.randn(B, Lf, 1, 1, device=x.device, dtype=Xf.real.dtype) * FREQ_PHASE_STD
        rot = torch.complex(torch.cos(phi), torch.sin(phi))
        Xf = Xf * rot

    # band op (stop or keep)
    keep = not (torch.rand(1).item() < FREQ_BAND_PROB)
    band_mask = _random_band_mask(B, Lf, device=x.device, keep=keep)  # real mask
    Xf = Xf * band_mask

    # spectral dropout
    if torch.rand(1).item() < FREQ_DROPOUT_PROB and FREQ_DROPOUT_FRAC > 0.0:
        k = max(1, int(FREQ_DROPOUT_FRAC * Lf))
        drop = torch.zeros(B, Lf, 1, 1, device=x.device)
        for b in range(B):
            idx = torch.randperm(Lf, device=x.device)[:k]
            drop[b, idx] = 1.0
        Xf = Xf * (1.0 - drop)

    # high/low shelf
    if torch.rand(1).item() < FREQ_SHELF_PROB:
        strength = torch.empty(1, device=x.device).uniform_(*FREQ_SHELF_STRENGTH).item()
        high = bool(torch.randint(0, 2, (1,), device=x.device).item())
        shelf = _shelf_curve(Lf, high=high, strength=strength, device=x.device)  # (1,Lf,1,1)
        Xf = Xf * shelf

    x_aug = torch.fft.irfft(Xf, n=T, dim=1)  # (B,T,N,F)
    x_aug = _variance_match(x_aug, x_ref=x)
    return x_aug

def make_view(x, mask_mode, mask_ratio, span_len, jitter_std, feat_drop):
    """
    Build one augmented view:
      - temporal mask (random/span),
      - optional feature dropout,
      - optional Gaussian jitter,
      - frequency-domain augmentations.
    """
    B, T, _, _ = x.shape
    m = make_temporal_mask(B, T, mode=mask_mode, ratio=mask_ratio, span_len=span_len, device=x.device)
    xv = apply_mask(x, m)
    xv = dropout_features(xv, p=feat_drop)
    xv = add_jitter(xv, std=jitter_std)
    xv = freq_augment(xv, prob=FREQ_AUG_PROB)
    return xv


# =========================
# Training loop
# =========================
def train_multitask_contrast(ds_train):
    # Build windows: X (nW, T, N, F), Y_fut (nW, H, N, F)
    X, Y_fut, Din, _ = prepare_multitask_windows(ds_train, seq_len=SEQ_LEN, horizon=H_PRE)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(Y_fut, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultiTaskCBEncoder(in_channels=Din, horizon=H_PRE).to(DEVICE)

    # Ensure trainable parts include LoRA + our adapters & heads
    for p in model.encoder.embed_proj.parameters():
        p.requires_grad = True
    for p in model.encoder.post_proj.parameters():
        p.requires_grad = True
    for p in model.recon_head.parameters():
        p.requires_grad = True
    for p in model.forecast_head.parameters():
        p.requires_grad = True
    for p in model.proj_head.parameters():
        p.requires_grad = True

    model.encoder.print_trainable()

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for xb, yb in tqdm(loader, desc=f"Pretrain Epoch {epoch}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)      # (B, T, N, F)
            yb = yb.to(DEVICE)      # (B, H, N, F)

            Bsz, T, _, _ = xb.shape

            # ---------- SSL mask for reconstruction ----------
            mask_ssl = make_temporal_mask(
                Bsz, T, mode=MASK_MODE, ratio=MASK_RATIO, span_len=SPAN_LEN, device=xb.device
            )
            xmasked = apply_mask(xb, mask_ssl)

            # ---------- Forward (reconstruction + forecasting) ----------
            opt.zero_grad()
            recon, fut = model(xmasked)  # recon: (B,T,N,F), fut: (B,H,N,F)

            loss_ssl  = masked_recon_loss(recon, xb, mask_ssl)
            loss_fore = forecast_loss(fut, yb)

            # ---------- Contrastive branch ----------
            if USE_CONTRASTIVE:
                x1 = make_view(
                    xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                    jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB
                )
                x2 = make_view(
                    xb, VIEW_MASK_MODE, VIEW_MASK_RATIO, VIEW_SPAN_LEN,
                    jitter_std=ADD_JITTER_STD, feat_drop=ADD_DROPOUT_PROB
                )
                z1 = model.pooled_projection(x1)  # (B, D_OUT)
                z2 = model.pooled_projection(x2)  # (B, D_OUT)
                loss_contrast = info_nce_loss(z1, z2, tau=TAU)
            else:
                loss_contrast = torch.tensor(0.0, device=xb.device)

            loss = W_SSL * loss_ssl + W_FORECAST * loss_fore + W_CONTRAST * loss_contrast

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(loader.dataset)
        took = time.time() - start
        print(f"[Pretrain] Epoch {epoch:02d} total={epoch_loss:.6f} "
              f"ssl={loss_ssl.item():.6f} fore={loss_fore.item():.6f} "
              f"contrast={loss_contrast.item():.6f} time={took:.2f}s")

    # Save ONLY the encoder wrapper weights (including LoRA adapters embedded in backbone)
    torch.save(
        {
            "cb_model_name": CB_MODEL_NAME,
            "state_dict": model.encoder.state_dict(),
            "d_out": D_OUT,
            "in_channels": Din,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
        },
        SAVE_PATH,
    )
    print(f"[Pretrain] Saved PEFT CB encoder wrapper to: {SAVE_PATH}")

    # Optionally export full LoRA-adapted backbone (HF style) for reuse
    try:
        os.makedirs(SAVE_ADAPTER_DIR, exist_ok=True)
        model.encoder.backbone.save_pretrained(SAVE_ADAPTER_DIR)
        print(f"[Pretrain] Saved LoRA adapter/backbone to: {SAVE_ADAPTER_DIR}")
    except Exception as e:
        print(f"[Pretrain] Adapter export skipped ({e}).")


# =========================
# Main
# =========================
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    train_multitask_contrast(ds_train)
    ds_train.close()

if __name__ == "__main__":
    main()
