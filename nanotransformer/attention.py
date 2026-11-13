import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanotransformer.config import TransformerConfig
from nanotransformer.rope import apply_rotary_emb


class KVCache:
    """Pre-allocated Key/Value cache for O(1) incremental token decoding."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.k_cache = torch.zeros(
            (batch_size, max_seq_len, n_kv_heads, head_dim), device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            (batch_size, max_seq_len, n_kv_heads, head_dim), device=device, dtype=dtype
        )

    def update(
        self, start_pos: int, xk: torch.Tensor, xv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _, _ = xk.shape
        end_pos = start_pos + seq_len
        self.k_cache[:bsz, start_pos:end_pos] = xk
        self.v_cache[:bsz, start_pos:end_pos] = xv
        return self.k_cache[:bsz, :end_pos], self.v_cache[:bsz, :end_pos]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats KV heads `n_rep` times to match Query heads: (B, T, H_kv, D) -> (B, T, H_q, D)."""
    if n_rep == 1:
        return x
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bsz, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)
    )


class CausalSelfAttention(nn.Module):
    """Grouped-Query Attention (GQA) with RoPE and optional autoregressive KVCache."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.kv_groups = config.kv_groups
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.dropout = config.dropout

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb(q, cos, sin, start_pos=start_pos)
        k = apply_rotary_emb(k, cos, sin, start_pos=start_pos)

        if kv_cache is not None:
            k, v = kv_cache.update(start_pos, k, v)

        k = repeat_kv(k, self.kv_groups)
        v = repeat_kv(v, self.kv_groups)

        # (B, H, T_q, D) and (B, H, T_kv, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if seq_len > 1:
            total_kv_len = k.shape[2]
            causal_mask = torch.full(
                (seq_len, total_kv_len),
                float("-inf"),
                device=x.device,
                dtype=scores.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1 + start_pos)
            scores = scores + causal_mask

        probs = F.softmax(scores.float(), dim=-1).type_as(x)
        if self.training and self.dropout > 0.0:
            probs = F.dropout(probs, p=self.dropout)

        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        return self.wo(out)
