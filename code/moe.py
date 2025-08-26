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


class Top1MoE(nn.Module):
    """Mixture-of-Experts block with sparsely-gated top-1 routing.

    The gating network selects a single expert for each input. A small
    load-balancing auxiliary loss encourages equal expert utilization.
    """

    def __init__(self, dim: int, num_experts: int = 2,
                 hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        hidden_dim = dim if hidden_dim is None else hidden_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),  # keep (B,T,N,D)->(B,T,N,D)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor, seed_logits: torch.Tensor | None = None,
                return_aux: bool = False):
        """Route inputs to a single expert selected by the gating network.

        Args:
            x: input tensor (..., dim)
            seed_logits: optional logits to add to gating scores
            return_aux: if True, also return load-balancing auxiliary loss
        """
        gate_logits = self.gate(x)
        if seed_logits is not None:
            gate_logits = gate_logits + seed_logits
        gate_prob = F.softmax(gate_logits, dim=-1)              # (..., E)
        top_idx = torch.argmax(gate_prob, dim=-1)                # (...,)
        dispatch = F.one_hot(top_idx, self.num_experts).type_as(gate_prob)  # (..., E)

        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (..., D, E)
        out = torch.sum(expert_outs * dispatch.unsqueeze(-2), dim=-1)             # (..., D)
        out = out * torch.sum(gate_prob * dispatch, dim=-1, keepdim=True)

        if not return_aux:
            return out

        # load balancing aux loss (Switch Transformer style)
        probs = gate_prob.view(-1, self.num_experts)
        loads = dispatch.view(-1, self.num_experts)
        importance = probs.sum(0) / probs.sum()
        load = loads.sum(0) / loads.sum()
        aux_loss = self.num_experts * (importance * load).sum()
        return out, aux_loss
