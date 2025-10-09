import torch
import torch.nn as nn
import torch.nn.functional as F
from stemgnn import STGNNForecaster

class ZeroInflatedStemGNN(nn.Module):
    """Two-stage outage forecaster with a classifier and regressor.

    The model first predicts whether an outage will occur (>0) using a
    classification head.  It then predicts the outage magnitude with a
    StemGNN forecaster, but only for locations/time steps classified as
    non-zero.  This design discourages false positive outages while still
    allowing accurate regression on real events.
    """

    def __init__(self, encoder: nn.Module, horizon: int, num_nodes: int, *, threshold: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.threshold = threshold
        # Output dimension of the encoder is needed for StemGNN input size.
        if hasattr(encoder, "d_model"):
            d_in = encoder.d_model
        else:
            raise AttributeError("Encoder must expose d_model attribute for embedding dim")

        # Binary classifier head predicts if outage > 0 for each horizon step.
        self.classifier = STGNNForecaster(d_in, horizon, num_nodes)
        # Regression head predicts magnitude conditioned on outage.
        self.regressor = STGNNForecaster(d_in, horizon, num_nodes)

    def forward(self, x):
        """Return (probability, magnitude, final_prediction)."""
        h = self.encoder(x)
        logits = self.classifier(h)           # (B, N, H)
        prob = torch.sigmoid(logits)
        magnitude = self.regressor(h)         # (B, N, H)
        # Mask regression output with classifier probability
        pred = torch.where(prob > self.threshold, magnitude, torch.zeros_like(magnitude))
        return prob, magnitude, pred


def two_stage_loss(prob, magnitude, target, *, zero_eps: float = 1e-8,
                   neg_weight: float = 5.0):
    """Compute combined loss for the two-stage model.

    Args:
        prob: classifier probability output (B, N, H)
        magnitude: regression output (B, N, H)
        target: ground truth outages (B, N, H)
        zero_eps: threshold for zero vs non-zero labels.
        neg_weight: additional weight for the negative class to penalize
            false positives (predicting outage when none).
    """
    label = (target > zero_eps).float()
    # weighted binary cross-entropy to penalise false positives
    weights = torch.where(label > 0, torch.ones_like(label), torch.full_like(label, neg_weight))
    cls_loss = F.binary_cross_entropy(prob, label, weight=weights)

    # regression loss only on true non-zero targets
    reg_mask = label > 0
    if reg_mask.any():
        reg_loss = F.mse_loss(magnitude[reg_mask], target[reg_mask])
    else:
        reg_loss = torch.tensor(0.0, device=target.device)

    return cls_loss + reg_loss

def train_zero_inflated(model, dataloader, optimizer, device):
    """Simple training loop for the zero-inflated model."""
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        prob, mag, _ = model(X)
        loss = two_stage_loss(prob, mag, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)
