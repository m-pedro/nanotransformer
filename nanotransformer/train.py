from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader

from nanotransformer.config import TransformerConfig
from nanotransformer.data import CosineWarmupScheduler
from nanotransformer.model import NanoTransformer


@dataclass
class TrainConfig:
    max_lr: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 10
    max_steps: int = 100
    batch_size: int = 8


class Trainer:
    """AdamW trainer with decoupled weight decay, gradient clipping, and checkpointing."""

    def __init__(
        self,
        model: NanoTransformer,
        train_config: TrainConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.cfg = train_config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # Apply weight decay only to 2D+ matrix weights (Linear/Embedding), not 1D RMSNorm scales
        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": train_config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=train_config.max_lr,
            betas=train_config.betas,
        )
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            max_lr=train_config.max_lr,
            min_lr=train_config.min_lr,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
        )

    def train_steps(self, dataset: torch.utils.data.Dataset) -> List[float]:
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.model.train()
        losses: List[float] = []
        step = 0

        while step < self.cfg.max_steps:
            for x, y in loader:
                if step >= self.cfg.max_steps:
                    break
                x = x.to(self.device)
                y = y.to(self.device)

                self.scheduler.step(step)
                self.optimizer.zero_grad(set_to_none=True)

                _, loss = self.model(x, targets=y)
                assert loss is not None
                loss.backward()

                if self.cfg.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                self.optimizer.step()
                losses.append(float(loss.item()))
                step += 1

        return losses

    @torch.no_grad()
    def evaluate_perplexity(self, dataset: torch.utils.data.Dataset) -> Dict[str, float]:
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0.0
        count = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            _, loss = self.model(x, targets=y)
            assert loss is not None
            total_loss += float(loss.item())
            count += 1
        mean_loss = total_loss / max(1, count)
        return {
            "loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 20.0)),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "model_config": asdict(self.model.config),
            "model_state": self.model.state_dict(),
        }
        torch.save(payload, Path(path))

    @staticmethod
    def load_checkpoint(
        path: str | Path, device: Optional[torch.device] = None
    ) -> NanoTransformer:
        payload = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
        cfg = TransformerConfig(**payload["model_config"])
        model = NanoTransformer(cfg)
        model.load_state_dict(payload["model_state"])
        if device is not None:
            model.to(device)
        return model
