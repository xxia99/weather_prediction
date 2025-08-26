import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    """Simple Mixture-of-Experts block with a softmax gating network."""
    def __init__(self, dim: int, out_dim: int, num_experts: int = 4,
                 hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        hidden_dim = dim if hidden_dim is None else hidden_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        gate = F.softmax(self.gate(x), dim=-1)  # (..., num_experts)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (..., out_dim, num_experts)
        out = torch.sum(expert_outs * gate.unsqueeze(-2), dim=-1)  # (..., out_dim)
        return out
