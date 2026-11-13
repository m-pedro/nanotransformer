# nanotransformer

A modern Llama-3 / GPT-style autoregressive Transformer built from scratch in Python and PyTorch. Implements Byte-Pair Encoding (`BPE`), Root Mean Square Normalization (`RMSNorm`), Rotary Positional Embeddings (`RoPE`), Grouped-Query Attention (`GQA`) with autoregressive `KV-Cache`, `SwiGLU` feed-forward layers, `top-k` / `top-p` nucleus sampling, and an `AdamW` training pipeline.

## 15-Day Incremental Engineering Roadmap (Month 4, Days 91–105)

- [x] **Day 91:** Initialize `nanotransformer` package, `TransformerConfig`, MIT license, and 15-day roadmap
- [ ] **Day 92:** Implement Byte-Pair Encoding (`BPE`) subword tokenizer from scratch (`nanotransformer/tokenizer.py`)
- [ ] **Day 93:** Implement Root Mean Square Layer Normalization (`RMSNorm`) (`nanotransformer/rmsnorm.py`)
- [ ] **Day 94:** Implement Rotary Positional Embeddings (`RoPE`) with frequency cache (`nanotransformer/rope.py`)
- [ ] **Day 95:** Implement scaled dot-product causal self-attention (`nanotransformer/attention.py`)
- [ ] **Day 96:** Upgrade attention to Grouped-Query Attention (`GQA`) with autoregressive `KV-Cache`
- [ ] **Day 97:** Implement `SwiGLU` gated feed-forward network (`nanotransformer/ffn.py`)
- [ ] **Day 98:** Assemble pre-norm `TransformerDecoderBlock` with residual connections (`nanotransformer/block.py`)
- [ ] **Day 99:** Build full `NanoTransformer` causal language model with weight tying (`nanotransformer/model.py`)
- [ ] **Day 100:** Implement temperature, `top-k`, and `top-p` (nucleus) autoregressive generation (`nanotransformer/generate.py`)
- [ ] **Day 101:** Build sliding-window next-token dataset loader and cosine warmup LR scheduler (`nanotransformer/data.py`)
- [ ] **Day 102:** Implement `AdamW` training loop with gradient clipping and checkpointing (`nanotransformer/train.py`)
- [ ] **Day 103:** Add convergence integration test verifying loss -> 0 on synthetic sequence memorization
- [ ] **Day 104:** Build `nanotransformer` CLI and KV-Cache speedup benchmark
- [ ] **Day 105:** Complete Llama-style Transformer architecture guide, tensor shape diagrams, and benchmarks
