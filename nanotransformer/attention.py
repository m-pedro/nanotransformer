import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanotransformer.config import TransformerConfig
from nanotransformer.rope import apply_rotary_emb


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product causal self-attention with RoPE."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.dropout = config.dropout

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim)

        q = apply_rotary_emb(q, cos, sin, start_pos=start_pos)
        k = apply_rotary_emb(k, cos, sin, start_pos=start_pos)

        # Transpose to (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if seq_len > 1:
            causal_mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=x.device, dtype=scores.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            scores = scores + causal_mask

        probs = F.softmax(scores.float(), dim=-1).type_as(x)
        if self.training and self.dropout > 0.0:
            probs = F.dropout(probs, p=self.dropout)

        out = torch.matmul(probs, v)  # (batch, n_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        return self.wo(out)
