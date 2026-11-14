import torch
from nanotransformer.config import TransformerConfig
from nanotransformer.data import CausalTextDataset
from nanotransformer.model import NanoTransformer
from nanotransformer.train import TrainConfig, Trainer


def test_trainer_reduces_loss_and_saves_checkpoint(tmp_path):
    torch.manual_seed(42)
    tokens = ([10, 20, 30, 40, 50, 60] * 25)
    ds = CausalTextDataset(tokens, seq_len=12, stride=6)

    cfg = TransformerConfig(
        vocab_size=128,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        max_seq_len=16,
    )
    model = NanoTransformer(cfg)
    tcfg = TrainConfig(max_lr=5e-3, min_lr=5e-4, warmup_steps=5, max_steps=30, batch_size=4)
    trainer = Trainer(model, tcfg, device=torch.device("cpu"))

    losses = trainer.train_steps(ds)
    assert losses[-1] < losses[0]

    metrics = trainer.evaluate_perplexity(ds)
    assert metrics["perplexity"] >= 1.0

    ckpt = tmp_path / "model.pt"
    trainer.save_checkpoint(ckpt)
    reloaded = Trainer.load_checkpoint(ckpt)
    x, _ = ds[0]
    logits1, _ = model.eval()(x.unsqueeze(0))
    logits2, _ = reloaded.eval()(x.unsqueeze(0))
    assert torch.allclose(logits1, logits2, atol=1e-6)
