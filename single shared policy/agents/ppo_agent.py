# File: agents/ppo_agent.py
"""PPO agent implementation for decentralized actors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def init_layer(layer: nn.Linear, std: float = 0.01) -> None:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)


@dataclass
class ActorConfig:
    """Configuration for actor network and PPO optimisation."""

    obs_dim: int
    action_dim: int
    hidden_dim: int = 128
    hidden_layers: int = 2
    lr: float = 3e-4
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


class ActorNetwork(nn.Module):
    """Multi-layer perceptron that outputs a categorical policy distribution."""

    def __init__(self, config: ActorConfig) -> None:
        super().__init__()
        layers = []
        input_dim = config.obs_dim
        for _ in range(config.hidden_layers):
            layer = nn.Linear(input_dim, config.hidden_dim)
            init_layer(layer)
            layers.extend([layer, nn.ReLU()])
            input_dim = config.hidden_dim
        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, config.action_dim)
        init_layer(self.policy_head, 0.01)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.body(observation) if len(self.body) > 0 else observation
        logits = self.policy_head(x)
        return logits


class PPOAgent:
    """Decentralised actor that participates in self-play training."""

    def __init__(self, config: ActorConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.actor = ActorNetwork(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Sample an action according to the current policy.

        Args:
            observation: Numpy array of shape (obs_dim,).
            deterministic: If True, take argmax instead of sampling.

        Returns:
            action (int), log_prob (float)
        """
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        logits = self.actor(obs_tensor)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return log probabilities and entropy for provided actions."""
        logits = self.actor(observations)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def state_dict(self):
        """Return underlying actor weights (used for saving)."""
        return self.actor.state_dict()

    def load_state_dict(self, state_dict):
        """Load weights into underlying actor network."""
        self.actor.load_state_dict(state_dict)

    def update(self, loss: torch.Tensor) -> None:
        """Backpropagate PPO loss and update parameters."""
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
