import torch
from nanotransformer.config import TransformerConfig
from nanotransformer.generate import generate
from nanotransformer.model import NanoTransformer


def test_greedy_generation_kv_cache_matches_uncached_decoding():
    torch.manual_seed(100)
    cfg = TransformerConfig(
        vocab_size=64,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        max_seq_len=32,
    )
    model = NanoTransformer(cfg).eval()
    prompt = torch.tensor([[1, 5, 12, 8]], dtype=torch.long)

    # Greedy decoding (temperature=0.0) is deterministic, so cached and uncached must match 100%
    out_cached = generate(model, prompt, max_new_tokens=8, temperature=0.0, use_kv_cache=True)
    out_uncached = generate(model, prompt, max_new_tokens=8, temperature=0.0, use_kv_cache=False)

    assert out_cached.shape == (1, 12)
    assert torch.equal(out_cached, out_uncached)

    # Verify stochastic top-k + top-p sampling produces valid token IDs
    sampled = generate(
        model, prompt, max_new_tokens=6, temperature=0.8, top_k=10, top_p=0.9, use_kv_cache=True
    )
    assert sampled.shape == (1, 10)
    assert int(sampled.min()) >= 0 and int(sampled.max()) < cfg.vocab_size
