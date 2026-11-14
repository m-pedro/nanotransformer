# nanotransformer

A modern Llama-3 / GPT-style autoregressive Transformer built from scratch in Python and PyTorch without `nn.Transformer` or Hugging Face wrappers. Implements Byte-Level Byte-Pair Encoding (`BPE`), Root Mean Square Layer Normalization (`RMSNorm`), Rotary Positional Embeddings (`RoPE`), Grouped-Query Attention (`GQA`) with autoregressive `KV-Cache`, `SwiGLU` gated feed-forward networks, `top-k` / `top-p` (nucleus) sampling, and an `AdamW` training pipeline.

---

## Repository Structure

```text
nanotransformer/
├── LICENSE                         # MIT License
├── README.md                       # Architecture guide, tensor shape flow & KV-Cache benchmarks
├── pyproject.toml                  # Package metadata & pytest configuration
├── nanotransformer/
│   ├── __init__.py                 # Public module exports
│   ├── config.py                   # TransformerConfig & GQA/RoPE validation
│   ├── tokenizer.py                # Byte-Level Byte-Pair Encoding (BPE) tokenizer
│   ├── rmsnorm.py                  # Root Mean Square Layer Normalization (RMSNorm)
│   ├── rope.py                     # Rotary Positional Embeddings (RoPE) & frequency cache
│   ├── attention.py                # Grouped-Query Attention (GQA) & O(1) KVCache
│   ├── ffn.py                      # SwiGLU gated feed-forward network (w_gate, w_up, w_down)
│   ├── block.py                    # Pre-RMSNorm TransformerDecoderBlock with residuals
│   ├── model.py                    # Full NanoTransformer causal LM with weight tying
│   ├── generate.py                 # Autoregressive decoding (temperature, top-k, top-p)
│   ├── data.py                     # Sliding-window CausalTextDataset & CosineWarmupScheduler
│   ├── train.py                    # AdamW Trainer, gradient clipping & checkpointing
│   └── cli.py                      # CLI runner (train-demo, bench-kv)
└── tests/
    ├── test_config.py              # Config validation & head group tests
    ├── test_tokenizer.py           # BPE compression, round-trip & persistence tests
    ├── test_rmsnorm.py             # Unit RMS & backward gradient verification
    ├── test_rope.py                # Orthogonal norm preservation & relative position tests
    ├── test_attention.py           # Causal mask & KVCache vs full-sequence equivalence
    ├── test_ffn.py                 # SwiGLU activation & gradient tests
    ├── test_block.py               # Pre-norm decoder block & incremental cache tests
    ├── test_model.py               # Weight tying, parameter counting & cross-entropy loss
    ├── test_generate.py            # Greedy cached/uncached parity & nucleus sampling
    ├── test_data.py                # Next-token shift & cosine warmup LR schedule tests
    ├── test_train.py               # AdamW step convergence & checkpoint reload tests
    └── test_overfit.py             # End-to-end sequence memorization (loss < 0.05) test
```

---

## Llama-Style Pre-Norm Transformer Architecture & Tensor Shapes

```text
  Input Token IDs: (B, T)
         │
         ▼
  [ Token Embedding (vocab_size -> D) ] ◄─── Weight Tied ───┐
         │  (B, T, D)                                       │
         ▼                                                  │
  ┌──────────────────────────────────────────────────────┐  │
  │ N x TransformerDecoderBlock                          │  │
  │                                                      │  │
  │   x ──► [ RMSNorm ]                                  │  │
  │            │                                         │  │
  │            ▼                                         │  │
  │   [ Grouped-Query Attention (GQA) ]                  │  │
  │     • Q: (B, T, H_q, D_h)   ──► RoPE(cos, sin)       │  │
  │     • K: (B, T, H_kv, D_h)  ──► RoPE ──► KVCache     │  │
  │     • V: (B, T, H_kv, D_h)  ───────────► KVCache     │  │
  │     • repeat_kv(H_kv -> H_q) & Causal Masked Softmax │  │
  │            │                                         │  │
  │   h = x + attn_out                                   │  │
  │            │                                         │  │
  │   h ──► [ RMSNorm ]                                  │  │
  │            │                                         │  │
  │            ▼                                         │  │
  │   [ SwiGLU FFN: (SiLU(h W_gate) * (h W_up)) W_down ] │  │
  │            │                                         │  │
  │   out = h + ffn_out                                  │  │
  └────────────┼─────────────────────────────────────────┘  │
               ▼                                            │
        [ Final RMSNorm ]                                   │
               │  (B, T, D)                                 │
               ▼                                            │
        [ LM Head Linear (D -> vocab_size) ] ───────────────┘
               │
               ▼
        Logits: (B, T, vocab_size) ──► Temperature / top-k / top-p
```

---

## Core Mathematical Primitives & Academic References

| Component | Formula / Mechanism | Reference Paper |
| :--- | :--- | :--- |
| **`RMSNorm`** | `x * rsqrt(mean(x^2) + eps) * weight` | Zhang & Sennrich, *Root Mean Square Layer Normalization* (NeurIPS 2019) |
| **`RoPE`** | 2D complex rotation by `m * theta^(-2i/d)` preserving relative offset `(m - n)` | Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) |
| **`GQA` + `KVCache`** | `H_kv` key/value heads shared across `H_q` query heads (`kv_groups = H_q / H_kv`) | Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (EMNLP 2023) |
| **`SwiGLU`** | `(SiLU(x W_gate) * (x W_up)) W_down` | Noam Shazeer, *GLU Variants Improve Transformer* (2020) |

---

## Autoregressive `KV-Cache` Benchmark (Apple Silicon)

Run locally via `python3 -m nanotransformer.cli bench-kv` (`4` layers, `128` dim, `8` query heads, `2` KV heads, generating `96` new tokens from a `16`-token prompt):

| Decoding Mode | Complexity per Step | Total Latency (96 tokens) | Throughput | Output Parity |
| :--- | :--- | ---: | ---: | :--- |
| **With `KVCache` (`use_kv_cache=True`)** | `O(1)` new K/V projections | ~36.8 ms | ~2,600 tokens/sec | Exact match |
| **Without `KVCache` (`use_kv_cache=False`)** | `O(T)` full sequence recompute | ~118.4 ms | ~810 tokens/sec | Baseline |

---

## Quickstart

### 1. Run the Full Test & Sequence Memorization Suite

```bash
source .venv/bin/activate
pytest -v
```

### 2. Run the End-to-End BPE + Transformer Training & Generation Demo

```bash
python3 -m nanotransformer.cli train-demo
```

### 3. Benchmark Autoregressive `KV-Cache` Speedup

```bash
python3 -m nanotransformer.cli bench-kv
```

---

## 15-Day Incremental Engineering Roadmap (Month 4, Days 91–105)

- [x] **Day 91:** Initialize `nanotransformer` package, `TransformerConfig`, MIT license, and 15-day roadmap
- [x] **Day 92:** Implement Byte-Pair Encoding (`BPE`) subword tokenizer from scratch (`nanotransformer/tokenizer.py`)
- [x] **Day 93:** Implement Root Mean Square Layer Normalization (`RMSNorm`) (`nanotransformer/rmsnorm.py`)
- [x] **Day 94:** Implement Rotary Positional Embeddings (`RoPE`) with frequency cache (`nanotransformer/rope.py`)
- [x] **Day 95:** Implement scaled dot-product causal self-attention (`nanotransformer/attention.py`)
- [x] **Day 96:** Upgrade attention to Grouped-Query Attention (`GQA`) with autoregressive `KV-Cache`
- [x] **Day 97:** Implement `SwiGLU` gated feed-forward network (`nanotransformer/ffn.py`)
- [x] **Day 98:** Assemble pre-norm `TransformerDecoderBlock` with residual connections (`nanotransformer/block.py`)
- [x] **Day 99:** Build full `NanoTransformer` causal language model with weight tying (`nanotransformer/model.py`)
- [x] **Day 100:** Implement temperature, `top-k`, and `top-p` (nucleus) autoregressive generation (`nanotransformer/generate.py`)
- [x] **Day 101:** Build sliding-window next-token dataset loader and cosine warmup LR scheduler (`nanotransformer/data.py`)
- [x] **Day 102:** Implement `AdamW` training loop with gradient clipping and checkpointing (`nanotransformer/train.py`)
- [x] **Day 103:** Add convergence integration test verifying loss -> 0 on synthetic sequence memorization (`tests/test_overfit.py`)
- [x] **Day 104:** Build `nanotransformer` CLI and KV-Cache speedup benchmark (`nanotransformer/cli.py`)
- [x] **Day 105:** Complete Llama-style Transformer architecture guide, tensor shape diagrams, and benchmarks
