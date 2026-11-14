import torch
from nanotransformer.config import TransformerConfig
from nanotransformer.data import CausalTextDataset
from nanotransformer.generate import generate
from nanotransformer.model import NanoTransformer
from nanotransformer.tokenizer import BPETokenizer
from nanotransformer.train import TrainConfig, Trainer


def test_nanotransformer_memorizes_synthetic_corpus_and_generates_exact_continuation():
    torch.manual_seed(7)
    pattern = "rust systems engineering and transformer attention mechanisms. "
    corpus = pattern * 24

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, target_vocab_size=300)
    token_ids = tokenizer.encode(corpus)

    seq_len = 16
    dataset = CausalTextDataset(token_ids, seq_len=seq_len, stride=4)

    cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=64,
    )
    model = NanoTransformer(cfg)
    tcfg = TrainConfig(
        max_lr=8e-3,
        min_lr=1e-3,
        warmup_steps=10,
        max_steps=120,
        batch_size=8,
    )
    trainer = Trainer(model, tcfg, device=torch.device("cpu"))
    losses = trainer.train_steps(dataset)

    assert losses[-1] < 0.05, f"expected memorized loss < 0.05, got {losses[-1]:.4f}"

    # Prompt with the first 5 tokens and verify greedy generation reproduces the next 10 tokens
    prompt_ids = token_ids[:5]
    expected_ids = token_ids[:15]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

    out_tensor = generate(
        model,
        prompt_tensor,
        max_new_tokens=10,
        temperature=0.0,
        use_kv_cache=True,
    )
    assert out_tensor[0].tolist() == expected_ids
