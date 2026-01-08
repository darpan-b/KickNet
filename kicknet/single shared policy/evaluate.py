"""Evaluation script for trained soccer policies."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from agents.ppo_agent import ActorConfig, PPOAgent
from configs.defaults import EvaluationConfig, TrainingConfig
from envs.soccer_env import SoccerEnv, SoccerEnvConfig


def run_episode(env: SoccerEnv, actor: PPOAgent, deterministic: bool, render: bool = False) -> Dict[str, float]:
    observations, _ = env.reset()
    possible_agents = env.possible_agents
    learning_team = "red"
    rewards_accumulator = {agent: 0.0 for agent in possible_agents}

    done = {agent: False for agent in possible_agents}
    while not all(done.values()):
        actions = {}
        for agent in possible_agents:
            if agent not in observations:
                continue
            if agent.startswith(learning_team):
                action, _ = actor.select_action(observations[agent], deterministic=deterministic)
            else:
                # Mirror match for evaluation
                action, _ = actor.select_action(observations[agent], deterministic=deterministic)
            actions[agent] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = {agent: terminations[agent] or truncations[agent] for agent in possible_agents}
        for agent, reward in rewards.items():
            rewards_accumulator[agent] += reward
        if render:
            env.render()
    return rewards_accumulator


def evaluate(model_path: Path, cfg: EvaluationConfig, env_cfg: SoccerEnvConfig) -> None:
    if not model_path.exists():
        print(f"Error: Model file not found at '{model_path}'")
        models_dir = Path("models")
        if models_dir.is_dir():
            print("\nAvailable models in 'models/' directory:")
            for f in sorted(models_dir.glob("*.pth")):
                print(f"  {f}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SoccerEnv(env_cfg)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).n
    actor_cfg = ActorConfig(obs_dim=obs_dim, action_dim=action_dim)
    actor = PPOAgent(actor_cfg, device=device)
    state_dict = torch.load(model_path, map_location=device)
    actor.load_state_dict(state_dict)

    all_rewards = []
    for episode in range(cfg.episodes):
        rewards = run_episode(env, actor, deterministic=True, render=cfg.render)
        episode_reward = np.mean([rewards[agent] for agent in env.possible_agents if agent.startswith("red")])
        print(f"Episode {episode + 1}: reward={episode_reward:.3f}")
        all_rewards.append(episode_reward)
    print(f"Average reward over {cfg.episodes} episodes: {np.mean(all_rewards):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained soccer policy")
    parser.add_argument(
        "--model", type=Path, default="models/soccer_actor_ep500.pth", help="Path to saved actor weights"
    )
    parser.add_argument("--episodes", type=int, default=EvaluationConfig.episodes)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvaluationConfig(episodes=args.episodes, render=args.render)
    env_cfg = SoccerEnvConfig(
        grid_height=TrainingConfig.grid_height,
        grid_width=TrainingConfig.grid_width,
        n_players_per_team=TrainingConfig.n_players_per_team,
        max_steps=TrainingConfig.max_steps,
    )
    evaluate(args.model, cfg, env_cfg)


if __name__ == "__main__":
    main()
