from typing import Tuple
import torch


def precompute_rope_cos_sin(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes (cos, sin) tables of shape (max_seq_len, head_dim // 2)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")

    exponents = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta ** exponents)
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Applies RoPE rotation to tensor `x` of shape (batch, seq_len, n_heads, head_dim)."""
    seq_len = x.shape[1]
    cos_slice = cos[start_pos : start_pos + seq_len].unsqueeze(0).unsqueeze(2)
    sin_slice = sin[start_pos : start_pos + seq_len].unsqueeze(0).unsqueeze(2)

    x_float = x.float()
    x_even = x_float[..., 0::2]
    x_odd = x_float[..., 1::2]

    rot_even = x_even * cos_slice - x_odd * sin_slice
    rot_odd = x_even * sin_slice + x_odd * cos_slice

    out = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
    return out.type_as(x)
