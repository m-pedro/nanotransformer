from typing import Optional
import torch
import torch.nn as nn

from nanotransformer.attention import CausalSelfAttention, KVCache
from nanotransformer.config import TransformerConfig
from nanotransformer.ffn import SwiGLUFFN
from nanotransformer.rmsnorm import RMSNorm


class TransformerDecoderBlock(nn.Module):
    """Pre-RMSNorm Transformer Decoder Block with residual connections."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        h = x + self.attn(
            self.attn_norm(x), cos, sin, start_pos=start_pos, kv_cache=kv_cache
        )
        out = h + self.ffn(self.ffn_norm(h))
        return out
