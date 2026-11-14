import torch
from nanotransformer.data import CausalTextDataset, CosineWarmupScheduler


def test_causal_dataset_shift_and_cosine_scheduler():
    tokens = list(range(50))
    ds = CausalTextDataset(tokens, seq_len=8, stride=4)
    assert len(ds) > 0

    x, y = ds[0]
    assert x.shape == (8,)
    assert y.shape == (8,)
    assert torch.equal(x[1:], y[:-1])

    param = torch.nn.Parameter(torch.randn(4, 4))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = CosineWarmupScheduler(opt, max_lr=1e-3, min_lr=1e-4, warmup_steps=5, max_steps=20)

    lr_start = sched.step(0)
    lr_peak = sched.step(4)
    lr_end = sched.step(20)

    assert lr_start < lr_peak
    assert abs(lr_peak - 1e-3) < 1e-9
    assert abs(lr_end - 1e-4) < 1e-9
