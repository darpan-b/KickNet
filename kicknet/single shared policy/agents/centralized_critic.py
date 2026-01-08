# File: agents/centralized_critic.py
"""Centralized critic module used by PPO in a CTDE setting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class CriticConfig:
    """Hyper-parameters controlling the centralized critic network."""

    input_dim: int
    hidden_dim: int = 128
    hidden_layers: int = 2


class CentralizedCritic(nn.Module):
    """Multi-layer perceptron estimating value of the joint state."""

    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        layers: Iterable[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return the scalar value estimate for a batch of joint states."""
        return self.model(state).squeeze(-1)
