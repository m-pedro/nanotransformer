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
