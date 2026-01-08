# File: training/train_selfplay.py
"""Training loop for self-play PPO with a centralized critic."""
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.centralized_critic import CentralizedCritic, CriticConfig
from agents.ppo_agent import ActorConfig, PPOAgent
from agents.selfplay_manager import SelfPlayManager
from configs.defaults import TrainingConfig
from envs.soccer_env_3v3 import SoccerEnv3v3
from utils.ppo_buffer import BufferConfig, RolloutBuffer


def build_joint_state(
    observations: Dict[str, np.ndarray], agent_order: List[str], obs_dim: int
) -> np.ndarray:
    """Concatenate observations into a joint state vector."""
    features: List[np.ndarray] = []
    zero = np.zeros(obs_dim, dtype=np.float32)
    for agent in agent_order:
        features.append(observations.get(agent, zero))
    return np.concatenate(features, axis=0)


def train(cfg: TrainingConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SoccerEnv3v3()  # no render during training

    possible_agents = env.possible_agents
    team0_ids = [agent for agent in possible_agents if int(agent.split("_")[1]) < 3]
    learner_ids = team0_ids

    obs_dim = env.observation_space[possible_agents[0]].shape[0]
    action_dim = env.action_space[possible_agents[0]].n

    actor_cfg = ActorConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=128,
        hidden_layers=2,
        lr=cfg.learning_rate,
        clip_param=cfg.clip_param,
        entropy_coef=cfg.entropy_coef,
        value_loss_coef=cfg.value_loss_coef,
        max_grad_norm=cfg.max_grad_norm,
    )
    critic_cfg = CriticConfig(
        input_dim=obs_dim * len(possible_agents), hidden_dim=256, hidden_layers=2
    )

    learner = PPOAgent(actor_cfg, device=device)
    critic = CentralizedCritic(critic_cfg).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate)

    buffer_cfg = BufferConfig(
        capacity=cfg.rollout_length * len(learner_ids),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        device=device,
    )
    buffer = RolloutBuffer(buffer_cfg, obs_dim=obs_dim, state_dim=critic_cfg.input_dim)
    selfplay_manager = SelfPlayManager(learner)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(cfg.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    for episode in range(1, cfg.episodes + 1):
        observations, _ = env.reset()
        buffer.clear()
        opponent_policy = selfplay_manager.get_opponent_policy()
        episode_reward = 0.0
        episode_steps = 0

        for t in range(cfg.rollout_length):
            joint_state = build_joint_state(observations, possible_agents, obs_dim)
            state_tensor = torch.as_tensor(
                joint_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                value = float(critic(state_tensor).item())

            actions: Dict[str, int] = {}
            log_probs: Dict[str, float] = {}
            for agent_id in list(observations.keys()):
                if agent_id in learner_ids:
                    action, log_prob = learner.select_action(observations[agent_id])
                    actions[agent_id] = action
                    log_probs[agent_id] = log_prob
                else:
                    action, _ = opponent_policy.select_action(
                        observations[agent_id], deterministic=False)
                    actions[agent_id] = action

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done_flags = {
                agent: terminations.get(agent, False) or truncations.get(agent, False)
                for agent in possible_agents
            }

            for agent_id in learner_ids:
                if agent_id in observations and agent_id in actions:
                    buffer.store(
                        observation=observations[agent_id],
                        action=actions[agent_id],
                        log_prob=log_probs[agent_id],
                        reward=rewards.get(agent_id, 0.0),
                        value=value,
                        done=done_flags.get(agent_id, False),
                        joint_state=joint_state,
                    )
                    episode_reward += rewards.get(agent_id, 0.0)

            episode_steps += 1
            observations = next_obs

            if not env.agents:
                break

        # Bootstrap value for final state
        if env.agents:
            final_state = build_joint_state(observations, possible_agents, obs_dim)
            final_tensor = torch.as_tensor(
                final_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                last_value = float(critic(final_tensor).item())
        else:
            last_value = 0.0

        buffer.compute_returns_and_advantages(last_value)

        # PPO update
        for _ in range(cfg.update_epochs):
            for batch in buffer.mini_batches(cfg.batch_size):
                new_log_probs, entropy = learner.evaluate_actions(
                    batch["observations"], batch["actions"]
                )
                ratio = torch.exp(new_log_probs - batch["log_probs"])
                surrogate1 = ratio * batch["advantages"]
                surrogate2 = torch.clamp(
                    ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param
                ) * batch["advantages"]
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                entropy_loss = entropy.mean()

                critic_values = critic(batch["joint_states"])
                value_loss = (batch["returns"] - critic_values).pow(2).mean()

                actor_loss = policy_loss - cfg.entropy_coef * entropy_loss
                learner.update(actor_loss)

                critic_optimizer.zero_grad()
                (cfg.value_loss_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                critic_optimizer.step()

        avg_reward = episode_reward / max(len(learner_ids), 1)
        writer.add_scalar("train/episode_reward", avg_reward, episode)
        writer.add_scalar("train/episode_length", episode_steps, episode)

        if episode % cfg.save_interval == 0:
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                learner.state_dict(),
                cfg.checkpoint_dir / f"soccer_actor_ep{episode}.pth",
            )
            torch.save(
                critic.state_dict(),
                cfg.checkpoint_dir / f"soccer_critic_ep{episode}.pth",
            )

        if episode % 10 == 0:
            selfplay_manager.update_opponent_policy()

        if episode % 10 == 0:
            print(
                f"Episode {episode}/{cfg.episodes}: "
                f"reward={avg_reward:.3f}, steps={episode_steps}"
            )

    writer.close()


def parse_args() -> argparse.Namespace:
    base_cfg = TrainingConfig()
    parser = argparse.ArgumentParser(description="Self-play PPO training entry point")
    parser.add_argument("--episodes", type=int, default=base_cfg.episodes)
    parser.add_argument("--rollout-length", type=int, default=base_cfg.rollout_length)
    parser.add_argument("--log-dir", type=Path, default=base_cfg.log_dir)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=base_cfg.checkpoint_dir
    )
    parser.add_argument("--save-interval", type=int, default=base_cfg.save_interval)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(
        episodes=args.episodes,
        rollout_length=args.rollout_length,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
    )
    train(cfg)


if __name__ == "__main__":
    main()
