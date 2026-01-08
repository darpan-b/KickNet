# File: configs/defaults.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    # PPO / training hyperparams
    episodes: int = 500
    rollout_length: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: int = 64

    # Logging / checkpoints
    log_dir: Path = Path("results/tensorboard")
    checkpoint_dir: Path = Path("models")
    save_interval: int = 100
