import torch
from nanotransformer.config import TransformerConfig
from nanotransformer.ffn import SwiGLUFFN


def test_swiglu_ffn_shape_and_gradients():
    torch.manual_seed(11)
    cfg = TransformerConfig(dim=32, hidden_dim=96, n_heads=4, n_kv_heads=2)
    ffn = SwiGLUFFN(cfg)

    x = torch.randn(2, 10, 32, requires_grad=True)
    y = ffn(x)
    assert y.shape == (2, 10, 32)

    y.sum().backward()
    assert x.grad is not None
    assert ffn.w_gate.weight.grad is not None
    assert ffn.w_up.weight.grad is not None
    assert ffn.w_down.weight.grad is not None
