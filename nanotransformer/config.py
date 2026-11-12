from dataclasses import dataclass
import torch


@dataclass
class TransformerConfig:
    vocab_size: int = 512
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    max_seq_len: int = 256
    hidden_dim: int = 384
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for Rotary Embeddings (RoPE)")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    @staticmethod
    def default_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
