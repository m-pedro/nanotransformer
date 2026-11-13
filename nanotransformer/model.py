from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanotransformer.attention import KVCache
from nanotransformer.block import TransformerDecoderBlock
from nanotransformer.config import TransformerConfig
from nanotransformer.rmsnorm import RMSNorm
from nanotransformer.rope import precompute_rope_cos_sin


class NanoTransformer(nn.Module):
    """Full autoregressive Llama/GPT-style Transformer Language Model."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        cos, sin = precompute_rope_cos_sin(
            config.head_dim, config.max_seq_len, theta=config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_kv_caches(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> List[KVCache]:
        dev = device or self.tok_embeddings.weight.device
        dtype = self.tok_embeddings.weight.dtype
        return [
            KVCache(
                batch_size=batch_size,
                max_seq_len=self.config.max_seq_len,
                n_kv_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                device=dev,
                dtype=dtype,
            )
            for _ in range(self.config.n_layers)
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_caches: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seq_len = tokens.shape
        if start_pos + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {start_pos + seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        h = self.tok_embeddings(tokens)
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h = layer(
                h,
                self.rope_cos,
                self.rope_sin,
                start_pos=start_pos,
                kv_cache=cache,
            )

        h = self.norm(h)
        logits = self.lm_head(h)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
