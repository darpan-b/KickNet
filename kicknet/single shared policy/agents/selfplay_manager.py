# File: agents/selfplay_manager.py
"""Self-play policy manager for PPO agents."""

from __future__ import annotations

import copy
from typing import Any


class SelfPlayManager:
    """Keeps a main (learning) policy and an opponent policy for self-play."""

    def __init__(self, initial_policy: Any) -> None:
        """
        Args:
            initial_policy: PPOAgent instance being trained.
        """
        self.main_policy = initial_policy
        # Create a separate opponent agent with identical architecture
        self.opponent_policy = copy.deepcopy(initial_policy)
        # Make sure weights start identical
        self.opponent_policy.load_state_dict(self.main_policy.state_dict())

    def get_main_policy(self):
        """Return the main policy (learner)."""
        return self.main_policy

    def get_opponent_policy(self):
        """Return the opponent policy (frozen copy of past main policy)."""
        return self.opponent_policy

    def update_opponent_policy(self) -> None:
        """Sync opponent policy weights with the main policy."""
        self.opponent_policy.load_state_dict(self.main_policy.state_dict())
