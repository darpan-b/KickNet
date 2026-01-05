"""Soccer environment for multi-agent reinforcement learning using PettingZoo."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from utils.renderer import Renderer


Action = int
Observation = np.ndarray


@dataclass
class SoccerEnvConfig:
    """Configuration container for :class:`SoccerEnv`.

    Attributes
    ----------
    grid_height:
        Number of rows in the rectangular pitch.
    grid_width:
        Number of columns in the rectangular pitch.
    n_players_per_team:
        Number of controlled agents per team. The total number of players in the
        environment is ``2 * n_players_per_team``.
    max_steps:
        Maximum number of simulation steps per episode before truncation.
    goal_reward:
        Reward delivered to the scoring team when the ball enters the opponent
        goal line.
    possession_reward:
        Dense shaping reward granted to agents that hold ball possession.
    step_penalty:
        Small penalty applied each step to encourage purposeful behaviour.
    foul_penalty:
        Penalty applied when agents collide with a teammate (discouraging
        overcrowding). Collisions with opponents result in the ball switching
        owners and a minor penalty.
    shared_reward:
        When ``True`` the entire team receives the same reward. Otherwise only
        the acting agent is rewarded.
    """

    grid_height: int = 5
    grid_width: int = 7
    n_players_per_team: int = 2
    max_steps: int = 200
    goal_reward: float = 1.0
    possession_reward: float = 0.02
    step_penalty: float = -0.005
    foul_penalty: float = -0.1
    shared_reward: bool = True


class SoccerEnv(ParallelEnv):
    """Simple grid-based multi-agent soccer environment.

    The implementation follows the PettingZoo parallel API. The environment is
    purposely lightweight so that it can be used as a pedagogical example for
    multi-agent reinforcement learning experiments without additional
    dependencies.
    """

    metadata = {"name": "soccer_v0", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(self, config: Optional[SoccerEnvConfig] = None) -> None:
        self.config = config or SoccerEnvConfig()
        self.possible_agents: List[str] = self._generate_agent_names()
        self.agents: List[str] = []
        self._action_space = spaces.Discrete(5)  # stay, up, down, left, right
        obs_dim = self._compute_observation_dim()
        low = np.full(obs_dim, -1.0, dtype=np.float32)
        high = np.full(obs_dim, 1.0, dtype=np.float32)
        self._observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._rng = random.Random()
        self.renderer = None

        # Mutable episode state
        self._positions: Dict[str, Tuple[int, int]] = {}
        self._ball_owner: Optional[str] = None
        self._ball_position: Tuple[int, int] = (0, 0)
        self._steps = 0

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Box:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng.seed(seed)
        self.agents = self.possible_agents[:]
        self._steps = 0
        self._positions = {}

        for agent in self.agents:
            self._positions[agent] = self._initial_position(agent)

        # Ball starts in the middle of the field with random possession
        self._ball_position = (
            self.config.grid_height // 2,
            self.config.grid_width // 2,
        )
        self._ball_owner = self._rng.choice(self.agents)
        self._positions[self._ball_owner] = self._ball_position

        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, Action]):
        if not self.agents:
            raise RuntimeError("Environment has terminated. Call reset().")

        self._steps += 1
        current_agents = list(self.agents)
        rewards = {agent: 0.0 for agent in current_agents}
        terminations = {agent: False for agent in current_agents}
        truncations = {agent: False for agent in current_agents}
        infos = {agent: {} for agent in current_agents}

        # Order actions by teams to make resolution deterministic
        for agent in current_agents:
            action = actions.get(agent, 0)
            self._resolve_movement(agent, action)

        # Reward shaping for ball possession
        if self._ball_owner is not None:
            owning_team = self._team(self._ball_owner)
            for agent in current_agents:
                if self._team(agent) == owning_team:
                    rewards[agent] += self.config.possession_reward

        # Detect collisions with teammates and opponents
        self._handle_collisions(rewards)

        # Check for goal condition
        scored_team = self._check_goal()
        if scored_team is not None:
            for agent in current_agents:
                if self._team(agent) == scored_team:
                    rewards[agent] += self.config.goal_reward
                else:
                    rewards[agent] -= self.config.goal_reward
            terminations = {agent: True for agent in current_agents}

        # Apply small step penalty to encourage faster play
        for agent in current_agents:
            rewards[agent] += self.config.step_penalty

        if self._steps >= self.config.max_steps:
            truncations = {agent: True for agent in current_agents}

        observations = {agent: self._observe(agent) for agent in current_agents}

        # Update active agent set according to done flags
        self.agents = [agent for agent in current_agents if not (terminations[agent] or truncations[agent])]

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        if self.renderer is None:
            self.renderer = Renderer(self.config.grid_height, self.config.grid_width)
        self.renderer.draw(self._positions, self._ball_position, self._ball_owner)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_agent_names(self) -> List[str]:
        agents = []
        for team in ("red", "blue"):
            for i in range(self.config.n_players_per_team):
                agents.append(f"{team}_{i}")
        return agents

    def _team(self, agent: str) -> str:
        return agent.split("_")[0]

    def _initial_position(self, agent: str) -> Tuple[int, int]:
        team = self._team(agent)
        row = self._rng.randrange(self.config.grid_height)
        if team == "red":
            col = self._rng.randrange(self.config.grid_width // 2)
        else:
            col = self._rng.randrange(self.config.grid_width // 2, self.config.grid_width)
        return row, col

    def _move(self, position: Tuple[int, int], action: Action) -> Tuple[int, int]:
        row, col = position
        if action == 1:  # up
            row = max(0, row - 1)
        elif action == 2:  # down
            row = min(self.config.grid_height - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)
        elif action == 4:  # right
            col = min(self.config.grid_width - 1, col + 1)
        return row, col

    def _resolve_movement(self, agent: str, action: Action) -> None:
        prev_position = self._positions[agent]
        new_position = self._move(prev_position, action)
        self._positions[agent] = new_position

        if self._ball_owner == agent:
            self._ball_position = new_position
        elif new_position == self._ball_position:
            self._ball_owner = agent

    def _handle_collisions(self, rewards: Dict[str, float]) -> None:
        # Count number of agents per cell
        occupancy: Dict[Tuple[int, int], List[str]] = {}
        for agent, position in self._positions.items():
            occupancy.setdefault(position, []).append(agent)

        for position, agents in occupancy.items():
            if len(agents) <= 1:
                continue
            teams = {self._team(agent) for agent in agents}
            if len(teams) == 1:
                # teammate collision -> penalty
                for agent in agents:
                    rewards[agent] += self.config.foul_penalty
            else:
                # contest for the ball: randomly assign possession to one team
                new_owner = self._rng.choice(agents)
                self._ball_owner = new_owner
                self._ball_position = position

    def _check_goal(self) -> Optional[str]:
        row, col = self._ball_position
        left_goal_col = 0
        right_goal_col = self.config.grid_width - 1

        if col == left_goal_col:
            return "blue"  # blue scored into red goal
        if col == right_goal_col:
            return "red"  # red scored into blue goal
        return None

    def _observe(self, agent: str) -> Observation:
        team = self._team(agent)
        own_row, own_col = self._positions[agent]
        ball_row, ball_col = self._ball_position

        def _norm_row(value: int) -> float:
            return 2.0 * value / (self.config.grid_height - 1) - 1.0

        def _norm_col(value: int) -> float:
            return 2.0 * value / (self.config.grid_width - 1) - 1.0

        observation: List[float] = [
            _norm_row(own_row),
            _norm_col(own_col),
            _norm_row(ball_row),
            _norm_col(ball_col),
            1.0 if self._ball_owner == agent else -1.0,
        ]

        teammates = [a for a in self.agents if self._team(a) == team and a != agent]
        opponents = [a for a in self.agents if self._team(a) != team]

        for mate in teammates:
            row, col = self._positions[mate]
            observation.extend([_norm_row(row), _norm_col(col)])
        # pad teammates if necessary
        while len(observation) < 5 + 2 * (self.config.n_players_per_team - 1):
            observation.extend([0.0, 0.0])

        for opponent in opponents:
            row, col = self._positions[opponent]
            observation.extend([_norm_row(row), _norm_col(col)])

        remaining = self.config.max_steps - self._steps
        observation.append(2.0 * remaining / self.config.max_steps - 1.0)

        return np.array(observation, dtype=np.float32)

    def _compute_observation_dim(self) -> int:
        teammates_dim = 2 * (self.config.n_players_per_team - 1)
        opponents_dim = 2 * self.config.n_players_per_team
        # own position (2) + ball position (2) + possession (1) + teammates + opponents + time
        return 2 + 2 + 1 + teammates_dim + opponents_dim + 1


def env(config: Optional[SoccerEnvConfig] = None) -> SoccerEnv:
    """Convenience factory for PettingZoo compatibility."""

    return SoccerEnv(config=config)
