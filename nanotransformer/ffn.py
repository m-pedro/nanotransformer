import torch
import torch.nn as nn
import torch.nn.functional as F

from nanotransformer.config import TransformerConfig


class SwiGLUFFN(nn.Module):
    """SwiGLU Gated Feed-Forward Network (Shazeer, 2020) used in Llama and Mistral."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_down = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.silu(self.w_gate(x)) * self.w_up(x)
        if self.training and self.dropout > 0.0:
            activated = F.dropout(activated, p=self.dropout)
        return self.w_down(activated)
