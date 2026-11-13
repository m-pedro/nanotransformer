import torch
from nanotransformer.config import TransformerConfig
from nanotransformer.model import NanoTransformer


def test_nanotransformer_forward_loss_and_weight_tying():
    torch.manual_seed(7)
    cfg = TransformerConfig(
        vocab_size=128,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        max_seq_len=32,
        tie_embeddings=True,
    )
    model = NanoTransformer(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()
    assert model.count_parameters() > 0

    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    targets = torch.randint(0, cfg.vocab_size, (2, 12))
    logits, loss = model(tokens, targets=targets)

    assert logits.shape == (2, 12, cfg.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0
