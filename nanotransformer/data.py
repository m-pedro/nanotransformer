import math
from typing import List, Tuple
import torch
from torch.utils.data import Dataset


class CausalTextDataset(Dataset):
    """Sliding-window dataset producing (input_ids, target_ids) shifted by 1 token."""

    def __init__(self, token_ids: List[int], seq_len: int, stride: int | None = None) -> None:
        if len(token_ids) <= seq_len:
            raise ValueError(
                f"token_ids length ({len(token_ids)}) must be strictly greater than seq_len ({seq_len})"
            )
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.starts = list(range(0, len(token_ids) - seq_len, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay learning rate schedule (Llama/GPT-3 style)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = max(0, warmup_steps)
        self.max_steps = max(1, max_steps)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps and self.warmup_steps > 0:
            return self.max_lr * float(step + 1) / float(self.warmup_steps)
        if step >= self.max_steps:
            return self.min_lr
        decay_steps = max(1, self.max_steps - self.warmup_steps)
        progress = float(step - self.warmup_steps) / float(decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.max_lr - self.min_lr)

    def step(self, step_num: int) -> float:
        lr = self.get_lr(step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
