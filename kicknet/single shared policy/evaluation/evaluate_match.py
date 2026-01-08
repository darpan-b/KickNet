# File: evaluation/evaluate_match.py
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from agents.ppo_agent import PPOAgent, ActorConfig
from envs.soccer_env_3v3 import SoccerEnv3v3


def infer_input_dim_from_state_dict(state_dict: dict) -> int:
    """
    Inspect state_dict to find a 2D weight tensor (e.g. first linear layer)
    and return its second dimension as the actor input dim.
    """
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        # Only consider 2D weight matrices
        if tensor.dim() == 2:
            # shape: [out_dim, in_dim]
            in_dim = tensor.shape[1]
            if in_dim > 0:
                return int(in_dim)
    raise RuntimeError("Unable to infer actor input dim from checkpoint; no 2D weight found.")


def evaluate(args):
    """
    Run continuous evaluation of a trained model in the soccer environment.
    One PPOAgent policy controls all players (both teams) for visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint first so we can infer actor input dim
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # load the checkpoint (cpu first to inspect shape)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # In many checkpoints you'll have nested dict (e.g., {'actor': state_dict, ...}).
    # Try to detect if the top-level keys are 'actor' or similar.
    # If so, try to extract the actor sub-state dict automatically.
    candidate = state_dict
    # common patterns to check
    for prefix_key in ("actor", "policy", "actor_network", "actor_net", "actor_body"):
        if isinstance(state_dict, dict) and prefix_key in state_dict and isinstance(state_dict[prefix_key], dict):
            candidate = state_dict[prefix_key]
            break

    # If candidate appears to be a wrapper with keys like 'state_dict', dig deeper
    if isinstance(candidate, dict) and "state_dict" in candidate and isinstance(candidate["state_dict"], dict):
        candidate = candidate["state_dict"]

    # Now candidate should be a mapping from parameter names to tensors; infer input dim
    try:
        inferred_input_dim = infer_input_dim_from_state_dict(candidate)
    except Exception as e:
        # fallback: try inferring from top-level state_dict if we drilled into a subdict and failed
        if candidate is not state_dict:
            try:
                inferred_input_dim = infer_input_dim_from_state_dict(state_dict)
            except Exception as e2:
                raise RuntimeError(
                    "Failed to infer actor input dim from checkpoint. "
                    "Checkpoint may be in an unexpected format."
                ) from e2
        else:
            raise RuntimeError(
                "Failed to infer actor input dim from checkpoint. "
                "Checkpoint may be in an unexpected format."
            ) from e

    # Create environment (with debug flag if requested)
    env = SoccerEnv3v3(render_mode="human" if args.render else None, debug=args.debug)

    base_agent_id = env.possible_agents[0]
    action_dim = env.action_space[base_agent_id].n

    # Build actor config using inferred input dim so shapes match checkpoint
    actor_cfg = ActorConfig(obs_dim=inferred_input_dim, action_dim=action_dim)

    # Instantiate agent on the correct device
    agent = PPOAgent(actor_cfg, device=device)

    # Now load actual state into the agent. If the checkpoint had a wrapper, attempt to find actor weights.
    load_target = candidate if candidate is not state_dict else state_dict
    try:
        agent.load_state_dict(load_target)
    except RuntimeError:
        # Try again with the original full state_dict (some checkpoints store without subkey)
        try:
            agent.load_state_dict(state_dict)
        except Exception as exc:
            # If still failing, re-raise with informative message
            raise RuntimeError(
                "Failed to load checkpoint into agent. The checkpoint parameter shapes "
                "do not match the constructed network. Tried both sub-dicts and full dict."
            ) from exc

    agent.actor.eval()

    try:
        while True:
            observations, _ = env.reset()
            done = False

            while not done:
                if args.render:
                    env.render()

                actions = {}
                for agent_id, obs in observations.items():
                    # obs is numpy array; agent.select_action may expect shape (obs_dim,)
                    action, _ = agent.select_action(obs, deterministic=True)
                    actions[agent_id] = action

                observations, _, terminations, truncations, _ = env.step(actions)
                done = any(terminations.values()) or any(truncations.values())

    except KeyboardInterrupt:
        print("Evaluation stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model for 3v3 Soccer."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained actor .pth file."
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the pygame UI (use --render to show the field)."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable environment debug prints (auto-pass/steal logs)."
    )
    args = parser.parse_args()
    evaluate(args)
