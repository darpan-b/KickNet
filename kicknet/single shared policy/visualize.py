"""Visualisation utilities for analysing trained soccer agents."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

from agents.ppo_agent import ActorConfig, PPOAgent
from envs.soccer_env import SoccerEnv, SoccerEnvConfig
from configs.defaults import TrainingConfig   # <-- load correct train config


def run_episode(
    env: SoccerEnv, actor: PPOAgent | None,
    deterministic: bool = True, render: bool = False
) -> Dict[str, List[Tuple[int, int]]]:

    observations, _ = env.reset()
    possible_agents = env.possible_agents
    trajectories: Dict[str, List[Tuple[int, int]]] = {agent: [] for agent in possible_agents}
    done = {agent: False for agent in possible_agents}

    while not all(done.values()):
        actions = {}
        for agent in possible_agents:
            if agent not in observations:
                continue

            if actor is not None:
                action, _ = actor.select_action(
                    observations[agent],
                    deterministic=deterministic
                )
            else:
                action = env.action_space(agent).sample()

            actions[agent] = action

        observations, rewards, terminations, truncations, _ = env.step(actions)
        done = {agent: terminations[agent] or truncations[agent] for agent in possible_agents}

        # store positions
        for agent in possible_agents:
            if agent in env._positions:
                trajectories[agent].append(env._positions[agent])

        if render:
            env.render()

    return trajectories


def plot_trajectories(
    trajectories: Dict[str, List[Tuple[int, int]]],
    grid_shape: Tuple[int, int]
) -> None:

    plt.figure(figsize=(6, 6))
    height, width = grid_shape

    for agent, coords in trajectories.items():
        if not coords:
            continue
        y = [p[0] for p in coords]
        x = [p[1] for p in coords]
        plt.plot(x, y, marker="o", label=agent)
        plt.text(x[0], y[0], f"{agent} start", fontsize=8)

    plt.xlim(-0.5, width - 0.5)
    plt.ylim(height - 0.5, -0.5)
    plt.grid(True)
    plt.legend()
    plt.xlabel("Field width")
    plt.ylabel("Field height")
    plt.title("Agent trajectories")
    plt.show()


def visualise(model_path: Path | None, render: bool = True) -> None:
    """Run one episode using the SAME env config as training."""

    # -----------------------------
    # 🔥 Use EXACT same env config as during training
    # -----------------------------
    train_cfg = TrainingConfig()

    env_cfg = SoccerEnvConfig(
        n_players_per_team=train_cfg.n_players_per_team,
        grid_height=train_cfg.grid_height,
        grid_width=train_cfg.grid_width,
        max_steps=train_cfg.max_steps,
    )

    env = SoccerEnv(env_cfg)

    actor: PPOAgent | None = None
    if model_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
        action_dim = env.action_space(env.possible_agents[0]).n

        actor_cfg = ActorConfig(
            obs_dim=obs_dim,
            action_dim=action_dim
        )

        actor = PPOAgent(actor_cfg, device=device)

        print(f"Loading model: {model_path}")
        state_dict = torch.load(model_path, map_location=device)

        try:
            actor.load_state_dict(state_dict)
        except Exception as e:
            print("\n❌ MODEL LOAD ERROR ❌")
            print(f"Model expects obs_dim={obs_dim}")
            print("The checkpoint you loaded has a different architecture.\n")
            raise e

    # run episode
    trajectories = run_episode(env, actor, render=render)

    if not render:
        plot_trajectories(trajectories, (env_cfg.grid_height, env_cfg.grid_width))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise a trained soccer agent")
    parser.add_argument("--model", type=Path, default=None, help="Path to an actor checkpoint (.pth)")
    parser.add_argument("--render", action="store_true", help="Render grid in terminal")
    return parser.parse_args()


def main() -> None:
    print("Usage: python visualize.py --model <path_to_model> --render")
    args = parse_args()

    model_path = args.model if args.model not in [None, Path("None")] else None

    visualise(model_path, args.render)


if __name__ == "__main__":
    main()
