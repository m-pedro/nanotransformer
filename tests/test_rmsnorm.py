import torch
from nanotransformer.rmsnorm import RMSNorm


def test_rmsnorm_unit_rms_and_gradients():
    torch.manual_seed(7)
    x = torch.randn(2, 16, 64, requires_grad=True)
    norm = RMSNorm(dim=64, eps=1e-6)
    y = norm(x)

    assert y.shape == (2, 16, 64)
    # With weight == 1, the root-mean-square across the last dimension must be ~1.0
    rms = torch.sqrt(y.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert norm.weight.grad is not None
