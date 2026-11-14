import sys
import time
import torch

from nanotransformer.config import TransformerConfig
from nanotransformer.data import CausalTextDataset
from nanotransformer.generate import generate
from nanotransformer.model import NanoTransformer
from nanotransformer.tokenizer import BPETokenizer
from nanotransformer.train import TrainConfig, Trainer


def run_train_demo() -> None:
    torch.manual_seed(42)
    corpus = (
        "Llama-style transformers combine RMSNorm, Rotary Positional Embeddings (RoPE), "
        "Grouped-Query Attention (GQA) with KV-Cache, and SwiGLU feed-forward networks. "
    ) * 30

    tok = BPETokenizer()
    tok.train(corpus, target_vocab_size=320)
    ids = tok.encode(corpus)

    ds = CausalTextDataset(ids, seq_len=24, stride=6)
    cfg = TransformerConfig(
        vocab_size=tok.vocab_size,
        dim=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=160,
        max_seq_len=64,
    )
    model = NanoTransformer(cfg)
    tcfg = TrainConfig(max_lr=6e-3, min_lr=5e-4, warmup_steps=10, max_steps=100, batch_size=8)
    trainer = Trainer(model, tcfg, device=torch.device("cpu"))

    t0 = time.perf_counter()
    losses = trainer.train_steps(ds)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    metrics = trainer.evaluate_perplexity(ds)

    prompt = "Llama-style transformers combine"
    prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out_ids = generate(model, prompt_ids, max_new_tokens=20, temperature=0.0, use_kv_cache=True)
    decoded = tok.decode(out_ids[0].tolist())

    print("=== nanotransformer End-to-End Training Demo ===")
    print(f"Parameters       : {model.count_parameters():,}")
    print(f"BPE Vocab Size   : {tok.vocab_size}")
    print(f"Initial Loss     : {losses[0]:.4f}")
    print(f"Final Loss       : {losses[-1]:.4f} (Perplexity: {metrics['perplexity']:.2f})")
    print(f"Train Wall Time  : {elapsed_ms:.1f} ms (100 steps)")
    print(f"Generated Sample : {decoded}")


def run_bench_kv() -> None:
    torch.manual_seed(7)
    cfg = TransformerConfig(
        vocab_size=512,
        dim=128,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        hidden_dim=384,
        max_seq_len=256,
    )
    model = NanoTransformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)
    gen_tokens = 96

    t0 = time.perf_counter()
    out_cached = generate(
        model, prompt, max_new_tokens=gen_tokens, temperature=0.0, use_kv_cache=True
    )
    cached_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    out_uncached = generate(
        model, prompt, max_new_tokens=gen_tokens, temperature=0.0, use_kv_cache=False
    )
    uncached_ms = (time.perf_counter() - t1) * 1000.0

    assert torch.equal(out_cached, out_uncached)
    speedup = uncached_ms / max(cached_ms, 1e-6)
    tok_per_sec = gen_tokens / (cached_ms / 1000.0)

    print("=== nanotransformer Autoregressive KV-Cache Benchmark ===")
    print(f"Model Config     : {cfg.n_layers}L, {cfg.dim}D, {cfg.n_heads}Q/{cfg.n_kv_heads}KV heads")
    print(f"Generated Tokens : {gen_tokens} (prompt=16)")
    print(f"With KV-Cache    : {cached_ms:.2f} ms ({tok_per_sec:.1f} tokens/sec)")
    print(f"Without KV-Cache : {uncached_ms:.2f} ms")
    print(f"KV-Cache Speedup : {speedup:.2f}x faster (exact token match verified)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "train-demo":
        run_train_demo()
    elif cmd == "bench-kv":
        run_bench_kv()
    else:
        print("Usage: python -m nanotransformer.cli <train-demo | bench-kv>")


if __name__ == "__main__":
    main()
