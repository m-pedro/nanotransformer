import torch
from nanotransformer.rope import apply_rotary_emb, precompute_rope_cos_sin


def test_rope_preserves_norm_and_satisfies_relative_position_invariance():
    torch.manual_seed(99)
    head_dim = 16
    cos, sin = precompute_rope_cos_sin(head_dim=head_dim, max_seq_len=64)

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    q_rot = apply_rotary_emb(q, cos, sin, start_pos=12)
    assert torch.allclose(q.norm(), q_rot.norm(), atol=1e-5)

    q_10 = apply_rotary_emb(q, cos, sin, start_pos=10)
    k_4 = apply_rotary_emb(k, cos, sin, start_pos=4)
    dot_a = (q_10 * k_4).sum()

    q_25 = apply_rotary_emb(q, cos, sin, start_pos=25)
    k_19 = apply_rotary_emb(k, cos, sin, start_pos=19)
    dot_b = (q_25 * k_19).sum()

    assert torch.allclose(dot_a, dot_b, atol=1e-5)
