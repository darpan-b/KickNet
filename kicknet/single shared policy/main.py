# File: main.py
import argparse
import sys
from pathlib import Path

from configs.defaults import TrainingConfig
from training.train_selfplay import train
from evaluation.evaluate_match import evaluate


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent 3v3 Soccer")
    subparsers = parser.add_subparsers(dest="command")

    # Training parser
    train_parser = subparsers.add_parser("train", help="Train the agents")
    base_cfg = TrainingConfig()
    train_parser.add_argument("--episodes", type=int, default=base_cfg.episodes)
    train_parser.add_argument(
        "--rollout-length", type=int, default=base_cfg.rollout_length
    )
    train_parser.add_argument(
        "--log-dir", type=str, default=str(base_cfg.log_dir)
    )
    train_parser.add_argument(
        "--checkpoint-dir", type=str, default=str(base_cfg.checkpoint_dir)
    )
    train_parser.add_argument(
        "--save-interval", type=int, default=base_cfg.save_interval
    )

    # Evaluation parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained actor .pth file"
    )

    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainingConfig(
            episodes=args.episodes,
            rollout_length=args.rollout_length,
            log_dir=Path(args.log_dir),
            checkpoint_dir=Path(args.checkpoint_dir),
            save_interval=args.save_interval,
        )
        train(cfg)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
