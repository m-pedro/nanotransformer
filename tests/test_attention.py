import torch
from nanotransformer.attention import CausalSelfAttention
from nanotransformer.config import TransformerConfig
from nanotransformer.rope import precompute_rope_cos_sin


def test_causal_mask_prevents_future_token_leakage():
    torch.manual_seed(123)
    cfg = TransformerConfig(dim=32, n_heads=4, n_kv_heads=4, max_seq_len=16)
    attn = CausalSelfAttention(cfg).eval()
    cos, sin = precompute_rope_cos_sin(cfg.head_dim, cfg.max_seq_len)

    x1 = torch.randn(1, 8, 32)
    x2 = x1.clone()
    # Perturb future tokens at indices 5, 6, 7
    x2[:, 5:, :] = torch.randn(1, 3, 32) * 100.0

    out1 = attn(x1, cos, sin)
    out2 = attn(x2, cos, sin)

    # Because of the causal triangular mask, outputs at positions 0..=4 must be identical!
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-6)
    # While outputs at positions 5..=7 must differ
    assert not torch.allclose(out1[:, 5:, :], out2[:, 5:, :])


def test_gqa_kv_cache_matches_full_sequence_forward_pass():
    from nanotransformer.attention import KVCache

    torch.manual_seed(42)
    cfg = TransformerConfig(dim=32, n_heads=4, n_kv_heads=2, max_seq_len=16)
    attn = CausalSelfAttention(cfg).eval()
    cos, sin = precompute_rope_cos_sin(cfg.head_dim, cfg.max_seq_len)

    x_full = torch.randn(1, 6, 32)
    out_full = attn(x_full, cos, sin, start_pos=0)

    # Incremental decoding: prefill 3 tokens, then decode 3 tokens one by one with KVCache
    cache = KVCache(
        batch_size=1,
        max_seq_len=cfg.max_seq_len,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
    )
    chunks = [attn(x_full[:, :3, :], cos, sin, start_pos=0, kv_cache=cache)]
    for pos in range(3, 6):
        step_out = attn(x_full[:, pos : pos + 1, :], cos, sin, start_pos=pos, kv_cache=cache)
        chunks.append(step_out)

    out_cached = torch.cat(chunks, dim=1)
    assert torch.allclose(out_full, out_cached, atol=1e-5)
