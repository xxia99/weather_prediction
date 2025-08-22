import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===================== Config =====================
SPLIT_DIR   = "data/"
OUT_DIR     = "results_seq/"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Data/windows
SEQ_LEN     = 24
BATCH_SIZE  = 32

# Training
EPOCHS        = 100
LR            = 1e-4
WEIGHT_DECAY  = 1e-3

# Masking (SSL)
MASK_RATIO = 0.50               # fraction of timesteps to mask per window
TEMPORAL_MASK_SHARED = True     # mask a timestep across all nodes & features

# Desired output embedding dim from the encoder (to match your downstream model)
D_OUT = 64                      # equals PATCH_D_MODEL in your forecaster

# ---- Chronos-Bolt + PEFT (LoRA) ----
CB_MODEL_NAME   = "/home2/caojia/weather_prediction/code/chronos-bolt-small"  # or a local path/variant
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

# Where to save the tuned encoder (load this into your forecaster to replace PatchTST)
SAVE_PATH = os.path.join(OUT_DIR, "pretrained_cb_encoder_peft.pt")
# Optional: save LoRA adapter directory in HF format
SAVE_ADAPTER_DIR = os.path.join(OUT_DIR, "cb_peft_adapter")


# ===================== Utils =====================
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

def build_windows(X_loc, y_loc, seq_len, horizon):
    """Build aligned windows for one location; horizon kept for signature parity."""
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return np.empty((0, seq_len, X_loc.shape[1]), dtype=float)
    Xw = []
    for i in range(N):
        Xw.append(X_loc[i:i+seq_len])
    return np.asarray(Xw, dtype=float)

def prepare_ssl_windows(ds_train, seq_len=24):
    """
    Returns:
      X: (num_windows, SEQ_LEN, N, Din)  with Din = 1 (outage) + F (weather)
      Din, scalers
    """
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, N)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, N, F)
    T, N, F = w.shape

    # scale features
    y_mu, y_sd = zfit(y.reshape(-1, 1))
    w_mu, w_sd = zfit(w.reshape(-1, F))
    y_sc = zapply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, N)     # (T, N)
    w_sc = zapply(w.reshape(-1, F), w_mu, w_sd).reshape(T, N, F)  # (T, N, F)

    Din = 1 + F
    X_list = []
    horizon_dummy = 1
    for node in range(N):
        x_loc_full = np.concatenate([y_sc[:, node:node+1], w_sc[:, node, :]], axis=1)  # (T, Din)
        Xw = build_windows(x_loc_full, y_sc[:, node], seq_len, horizon_dummy)          # (nW, SEQ_LEN, Din)
        if len(Xw):
            X_list.append(Xw)

    if not X_list:
        raise RuntimeError("No SSL windows could be built (check SEQ_LEN vs data length).")

    # Stack across nodes → (nW, SEQ_LEN, N, Din)
    X = np.stack(X_list, axis=2)
    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    return X, Din, scalers


# ===================== Chronos-Bolt PEFT Encoder =====================
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
    Inspect Linear submodules and pick target substrings that actually exist
    in this backbone. Preference: explicit proj names -> attention -> encoder linears -> 'linear'.
    """
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    # Common patterns across HF backbones (incl. T5)
    candidates = [
        # LLaMA-style
        "q_proj", "k_proj", "v_proj", "o_proj",
        # NeoX/Falcon-style
        "query_key_value", "out_proj", "in_proj",
        # Diffusers & misc.
        "to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out",
        # T5-style leaf names
        "q", "k", "v", "o", "wi", "wo", "wi_0", "wi_1",
        # Some custom
        "Wqkv", "Wo", "Wq", "Wk", "Wv",
    ]
    present = [s for s in candidates if any(s in n for n in linear_names)]

    # If none matched, try attention-related linears by leaf names
    if not present:
        attn_linear = [n for n in linear_names if ("attn" in n.lower() or "attention" in n.lower())]
        if attn_linear:
            leaves = list({n.split(".")[-1] for n in attn_linear})
            present = leaves[:6]

    # Still nothing? Use encoder block linears' leaf names
    if not present:
        enc_linear = [n for n in linear_names if (".encoder." in n or "encoder.layers" in n or ".layers." in n)]
        if enc_linear:
            present = list({n.split(".")[-1] for n in enc_linear})

    # Absolute fallback
    if not present:
        present = ["linear"]

    # Unique preserve order
    seen, final = set(), []
    for s in present:
        if s not in seen:
            seen.add(s)
            final.append(s)
    return final, linear_names


class ChronosBoltEncoderPEFT(nn.Module):
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

        # Load CB backbone (often T5-based)
        self.config = AutoConfig.from_pretrained(cb_model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            cb_model_name,
            trust_remote_code=True,
            output_hidden_states=True
        )

        # Discover LoRA targets
        targets, linear_names = _discover_lora_targets(self.backbone)
        print(f"[PEFT] Discovered LoRA target substrings: {targets}")

        # Inject LoRA
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
            print(f"[PEFT] Warning: {e}")
            print("[PEFT] Falling back to target_modules=['linear']")
            lcfg.target_modules = ["linear"]
            self.backbone = get_peft_model(self.backbone, lcfg)

        # Determine hidden size
        hidden_dim = getattr(self.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.config, "d_model", None)
        if hidden_dim is None:
            raise RuntimeError("Cannot determine Chronos hidden size from config.")

        # Map multivariate features -> hidden_dim per step (so we can use inputs_embeds)
        self.embed_proj = nn.Linear(in_channels, hidden_dim)

        # Post projection to desired D
        self.post_proj = nn.Linear(hidden_dim, d_out)

    def forward(self, x):
        """
        x: (B, T, N, F)  -> returns (B, T, N, D_OUT)
        Strategy:
          - Project features directly to model hidden size (inputs_embeds)
          - Call encoder only (works for T5-style backbones)
        """
        B, T, N, F = x.shape
        assert F == self.in_channels, f"CB encoder expected F={self.in_channels}, got {F}"

        # inputs_embeds: (B*N, T, hidden_dim)
        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, F)   # (B*N, T, F)
        emb = self.embed_proj(x_bn)                         # (B*N, T, H)

        # Full attention mask (no padding)
        attn_mask = torch.ones(B * N, T, dtype=torch.long, device=emb.device)

        # Prefer calling encoder directly if available (e.g., T5Model.encoder)
        if hasattr(self.backbone, "encoder") and callable(self.backbone.encoder):
            out = self.backbone.encoder(
                inputs_embeds=emb,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        else:
            # Fall back to calling the backbone with inputs_embeds
            out = self.backbone(
                inputs_embeds=emb,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # Get sequence hidden states
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            h_bn = out.last_hidden_state  # (B*N, T', H)
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            h_bn = out.hidden_states[-1]
        else:
            if isinstance(out, (tuple, list)) and len(out) > 0:
                h_bn = out[0]
            else:
                raise RuntimeError("Backbone forward did not return hidden states.")

        # Ensure time dim matches T
        if h_bn.size(1) > T:
            h_bn = h_bn[:, :T, :]
        elif h_bn.size(1) < T:
            pad_len = T - h_bn.size(1)
            pad = torch.zeros(h_bn.size(0), pad_len, h_bn.size(2), device=h_bn.device, dtype=h_bn.dtype)
            h_bn = torch.cat([h_bn, pad], dim=1)

        # Project to D_OUT and reshape back to (B, T, N, D_OUT)
        h_bn = self.post_proj(h_bn)                # (B*N, T, D_OUT)
        h = h_bn.view(B, N, T, self.d_out).permute(0, 2, 1, 3)
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


# ===================== SSL Head (masked-step prediction) =====================
class CBMaskedStepPredictor(nn.Module):
    """
    Chronos-Bolt encoder (with LoRA) + linear head to reconstruct inputs.
    Loss is computed only on masked timesteps.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = ChronosBoltEncoderPEFT(in_channels=in_channels, d_out=D_OUT)
        self.head = nn.Linear(D_OUT, in_channels)

    def forward(self, x):
        # x: (B, T, N, F)
        h = self.encoder(x)    # (B, T, N, D_OUT)
        y = self.head(h)       # (B, T, N, F)
        return y


# ===================== Training (masked-step SSL) =====================
def masked_loss(pred, target, mask):
    """
    pred/target: (B, T, N, F)
    mask: (B, T, 1, 1) -> 1 where masked (to be predicted)
    """
    m = mask.expand_as(pred)
    denom = m.sum()
    if denom.item() < 1.0:
        return torch.mean((pred - target) ** 2)
    return ((pred - target) ** 2 * m).sum() / denom


def train_ssl(ds_train):
    # Build windows
    X, Din, _ = prepare_ssl_windows(ds_train, seq_len=SEQ_LEN)  # (nW, T, N, Din)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = CBMaskedStepPredictor(in_channels=Din).to(DEVICE)

    # PEFT sets only LoRA params trainable in backbone; also train our adapters & head
    for p in model.encoder.embed_proj.parameters():
        p.requires_grad = True
    for p in model.encoder.post_proj.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

    model.encoder.print_trainable()

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for (xb,) in tqdm(dl, desc=f"SSL Epoch {epoch}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)  # (B, T, N, F)

            # Temporal mask per sample
            if TEMPORAL_MASK_SHARED:
                with torch.no_grad():
                    B, T, _, _ = xb.shape
                    mask = (torch.rand(B, T, 1, 1, device=xb.device) < MASK_RATIO).float()
                    xmasked = xb * (1.0 - mask)
            else:
                with torch.no_grad():
                    m_nf = (torch.rand_like(xb[:, :, :, :1]) < MASK_RATIO).float()  # (B, T, N, 1)
                    xmasked = xb * (1.0 - m_nf)
                    mask = (m_nf.mean(dim=2, keepdim=True) > 0).float()

            opt.zero_grad()
            yhat = model(xmasked)               # (B, T, N, F)
            loss = masked_loss(yhat, xb, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        print(f"[SSL] Epoch {epoch:03d}  Loss={epoch_loss:.6f}  Time={time.time()-start:.2f}s")

    # ===== Save encoder wrapper weights (including LoRA) =====
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
    print(f"[SSL] Saved PEFT CB encoder wrapper to: {SAVE_PATH}")

    # Optionally also save HF-style LoRA adapters (if supported by the backbone)
    try:
        os.makedirs(SAVE_ADAPTER_DIR, exist_ok=True)
        model.encoder.backbone.save_pretrained(SAVE_ADAPTER_DIR)
        print(f"[SSL] Saved LoRA adapter to: {SAVE_ADAPTER_DIR}")
    except Exception as e:
        print(f"[SSL] Skipped adapter export ({e}).")


# ===================== Main =====================
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    train_ssl(ds_train)
    ds_train.close()

if __name__ == "__main__":
    main()
