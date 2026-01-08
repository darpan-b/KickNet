# File: utils/ppo_buffer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Dict, Any

import numpy as np
import torch


@dataclass
class BufferConfig:
    capacity: int
    gamma: float
    gae_lambda: float
    device: torch.device | str = "cpu"


class RolloutBuffer:
    """On-policy buffer with GAE for PPO."""

    def __init__(self, config: BufferConfig, obs_dim: int, state_dim: int) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.obs_dim = obs_dim
        self.state_dim = state_dim

        self.clear()

    def clear(self) -> None:
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.joint_states = []

    def store(
        self,
        observation,
        action,
        log_prob,
        reward,
        value,
        done,
        joint_state,
    ) -> None:
        self.observations.append(np.array(observation, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
        self.joint_states.append(np.array(joint_state, dtype=np.float32))

    def _to_tensors(self) -> Dict[str, torch.Tensor]:
        obs = torch.as_tensor(self.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions, dtype=torch.long, device=self.device)
        log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(self.values, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=self.device)
        joint_states = torch.as_tensor(
            self.joint_states, dtype=torch.float32, device=self.device
        )
        return {
            "observations": obs,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "values": values,
            "dones": dones,
            "joint_states": joint_states,
        }

    def compute_returns_and_advantages(self, last_value: float) -> None:
        data = self._to_tensors()
        rewards = data["rewards"]
        values = data["values"]
        dones = data["dones"]

        gamma = self.config.gamma
        lam = self.config.gae_lambda

        T = rewards.shape[0]
        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.tensor(last_value, device=self.device)
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        self.advantages = advantages
        self.returns = returns
        # cache tensors for mini_batches
        self._cached = data

    def mini_batches(self, batch_size: int) -> Iterator[Dict[str, Any]]:
        data = self._cached
        obs = data["observations"]
        actions = data["actions"]
        log_probs = data["log_probs"]
        advantages = self.advantages
        returns = self.returns
        joint_states = data["joint_states"]

        N = obs.shape[0]
        indices = np.arange(N)
        np.random.shuffle(indices)

        for start in range(0, N, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                "observations": obs[batch_idx],
                "actions": actions[batch_idx],
                "log_probs": log_probs[batch_idx],
                "advantages": advantages[batch_idx],
                "returns": returns[batch_idx],
                "joint_states": joint_states[batch_idx],
            }
