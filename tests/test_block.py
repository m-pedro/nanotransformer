import torch
from nanotransformer.attention import KVCache
from nanotransformer.block import TransformerDecoderBlock
from nanotransformer.config import TransformerConfig
from nanotransformer.rope import precompute_rope_cos_sin


def test_decoder_block_residual_and_cached_equivalence():
    torch.manual_seed(21)
    cfg = TransformerConfig(dim=32, n_heads=4, n_kv_heads=2, hidden_dim=64, max_seq_len=16)
    block = TransformerDecoderBlock(cfg).eval()
    cos, sin = precompute_rope_cos_sin(cfg.head_dim, cfg.max_seq_len)

    x = torch.randn(1, 5, 32)
    full_out = block(x, cos, sin, start_pos=0)

    cache = KVCache(1, cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim)
    parts = [block(x[:, :2, :], cos, sin, start_pos=0, kv_cache=cache)]
    for pos in range(2, 5):
        parts.append(block(x[:, pos : pos + 1, :], cos, sin, start_pos=pos, kv_cache=cache))

    cached_out = torch.cat(parts, dim=1)
    assert torch.allclose(full_out, cached_out, atol=1e-5)
