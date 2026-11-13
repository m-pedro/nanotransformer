from typing import Optional
import torch
import torch.nn.functional as F

from nanotransformer.model import NanoTransformer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Samples next token IDs from (batch, vocab_size) logits using temperature, top-k, and top-p."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, scaled_logits.size(-1))
        top_vals, _ = torch.topk(scaled_logits, k, dim=-1)
        threshold = top_vals[:, -1].unsqueeze(-1)
        scaled_logits = torch.where(
            scaled_logits < threshold,
            torch.full_like(scaled_logits, float("-inf")),
            scaled_logits,
        )

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Mask tokens with cumulative probability above top_p (shifting right by 1 to always keep top-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        scaled_logits = torch.zeros_like(scaled_logits).scatter(
            dim=-1, index=sorted_indices, src=sorted_logits
        )

    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: NanoTransformer,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """Autoregressively generates `max_new_tokens` tokens with optional KVCache acceleration."""
    model.eval()
    bsz, prompt_len = prompt_tokens.shape
    tokens = prompt_tokens.clone()

    if not use_kv_cache:
        for _ in range(max_new_tokens):
            ctx = tokens[:, -model.config.max_seq_len :]
            logits, _ = model(ctx, start_pos=0, kv_caches=None)
            next_tok = sample_next_token(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            tokens = torch.cat([tokens, next_tok], dim=1)
            if eos_id is not None and torch.all(next_tok == eos_id):
                break
        return tokens

    kv_caches = model.init_kv_caches(batch_size=bsz, device=tokens.device)

    # Phase 1: Prefill the prompt into the KVCache
    logits, _ = model(tokens, start_pos=0, kv_caches=kv_caches)
    next_tok = sample_next_token(
        logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
    )
    tokens = torch.cat([tokens, next_tok], dim=1)

    # Phase 2: Decode single tokens incrementally using cached Keys and Values
    curr_pos = prompt_len
    for _ in range(max_new_tokens - 1):
        if eos_id is not None and torch.all(next_tok == eos_id):
            break
        if curr_pos >= model.config.max_seq_len:
            break
        logits, _ = model(next_tok, start_pos=curr_pos, kv_caches=kv_caches)
        next_tok = sample_next_token(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )
        tokens = torch.cat([tokens, next_tok], dim=1)
        curr_pos += 1

    return tokens
