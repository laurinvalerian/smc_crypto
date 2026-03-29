"""
Train DQN Exit Manager on historical trade episodes.

Usage:
    python3 -m rl_dqn.train_dqn [--episodes-dir data/rl_training]
                                 [--classes crypto stocks]
                                 [--total-timesteps 500000]
                                 [--output models/dqn_exit_manager]
                                 [--walk-forward]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Walk-forward windows (3-month train, 1-month OOS)
WALK_FORWARD_WINDOWS = [
    {"train_start": "2024-01-01", "train_end": "2024-03-31",
     "eval_start": "2024-04-01", "eval_end": "2024-04-30"},
    {"train_start": "2024-01-01", "train_end": "2024-06-30",
     "eval_start": "2024-07-01", "eval_end": "2024-07-31"},
    {"train_start": "2024-01-01", "train_end": "2024-09-30",
     "eval_start": "2024-10-01", "eval_end": "2024-10-31"},
]


def _get_dqn_kwargs() -> dict:
    """Standard DQN hyperparameters for exit management."""
    return dict(
        policy="MlpPolicy",
        policy_kwargs=dict(net_arch=[128, 128]),
        buffer_size=500_000,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        optimize_memory_usage=True,
        learning_starts=1000,
        verbose=1,
    )


def _evaluate_model(model, env, n_episodes: int = 100) -> dict:
    """Run evaluation episodes and return stats."""
    rewards = []
    lengths = []
    exit_reasons: dict[str, int] = {}

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
        reason = info.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        "avg_reward": float(sum(rewards) / max(len(rewards), 1)),
        "std_reward": float(
            (sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards)
             / max(len(rewards), 1)) ** 0.5
        ) if rewards else 0.0,
        "avg_length": float(sum(lengths) / max(len(lengths), 1)),
        "total_episodes": len(rewards),
        "exit_reasons": exit_reasons,
    }


def train_single(
    episodes_dir: str,
    asset_classes: list[str] | None,
    total_timesteps: int,
    output_path: str,
    walk_forward_start: str | None = None,
    walk_forward_end: str | None = None,
) -> None:
    """Train a single DQN model."""
    try:
        from stable_baselines3 import DQN
    except ImportError:
        logger.error(
            "stable-baselines3 is required for training. "
            "Install with: pip install 'stable-baselines3[extra]'"
        )
        sys.exit(1)

    from rl_dqn.trade_env import TradeManagementEnv

    logger.info("Creating environment (dir=%s, classes=%s)", episodes_dir, asset_classes)
    env = TradeManagementEnv(
        episodes_dir=episodes_dir,
        asset_classes=asset_classes,
        walk_forward_start=walk_forward_start,
        walk_forward_end=walk_forward_end,
    )
    logger.info("Environment created with %d episodes", len(env.episodes))

    dqn_kwargs = _get_dqn_kwargs()
    model = DQN(env=env, **dqn_kwargs)

    logger.info("Starting training for %d timesteps ...", total_timesteps)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    logger.info("Training completed in %.1f seconds", elapsed)

    # Evaluate
    eval_stats = _evaluate_model(model, env)
    logger.info(
        "Eval: avg_reward=%.4f (+/-%.4f), avg_length=%.1f, episodes=%d",
        eval_stats["avg_reward"],
        eval_stats["std_reward"],
        eval_stats["avg_length"],
        eval_stats["total_episodes"],
    )
    logger.info("Exit reasons: %s", eval_stats["exit_reasons"])

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    logger.info("Model saved to %s", out)


def train_walk_forward(
    episodes_dir: str,
    asset_classes: list[str] | None,
    total_timesteps: int,
    output_path: str,
) -> None:
    """Walk-forward training: train on expanding window, evaluate on next OOS."""
    try:
        from stable_baselines3 import DQN
    except ImportError:
        logger.error(
            "stable-baselines3 is required for training. "
            "Install with: pip install 'stable-baselines3[extra]'"
        )
        sys.exit(1)

    from rl_dqn.trade_env import TradeManagementEnv

    for i, window in enumerate(WALK_FORWARD_WINDOWS):
        logger.info(
            "=== Walk-Forward Window %d/%d ===", i + 1, len(WALK_FORWARD_WINDOWS),
        )
        logger.info(
            "Train: %s -> %s | Eval: %s -> %s",
            window["train_start"], window["train_end"],
            window["eval_start"], window["eval_end"],
        )

        # Train env
        train_env = TradeManagementEnv(
            episodes_dir=episodes_dir,
            asset_classes=asset_classes,
            walk_forward_start=window["train_start"],
            walk_forward_end=window["train_end"],
        )
        logger.info("Train episodes: %d", len(train_env.episodes))

        # Eval env
        eval_env = TradeManagementEnv(
            episodes_dir=episodes_dir,
            asset_classes=asset_classes,
            walk_forward_start=window["eval_start"],
            walk_forward_end=window["eval_end"],
        )
        logger.info("Eval episodes: %d", len(eval_env.episodes))

        dqn_kwargs = _get_dqn_kwargs()
        model = DQN(env=train_env, **dqn_kwargs)

        t0 = time.time()
        model.learn(total_timesteps=total_timesteps)
        elapsed = time.time() - t0
        logger.info("Window %d training done in %.1f s", i + 1, elapsed)

        # Evaluate on OOS
        eval_stats = _evaluate_model(model, eval_env)
        logger.info(
            "Window %d OOS: avg_reward=%.4f (+/-%.4f), avg_length=%.1f",
            i + 1,
            eval_stats["avg_reward"],
            eval_stats["std_reward"],
            eval_stats["avg_length"],
        )
        logger.info("Exit reasons: %s", eval_stats["exit_reasons"])

        # Save per-window model
        window_path = Path(output_path).with_stem(
            f"{Path(output_path).stem}_w{i + 1}"
        )
        model.save(str(window_path))
        logger.info("Window %d model saved to %s", i + 1, window_path)

    # Final model: train on all data
    logger.info("=== Final model: training on all available data ===")
    full_env = TradeManagementEnv(
        episodes_dir=episodes_dir,
        asset_classes=asset_classes,
    )
    logger.info("Full dataset: %d episodes", len(full_env.episodes))

    dqn_kwargs = _get_dqn_kwargs()
    final_model = DQN(env=full_env, **dqn_kwargs)
    final_model.learn(total_timesteps=total_timesteps)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(str(out))
    logger.info("Final model saved to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DQN Exit Manager on historical trade episodes",
    )
    parser.add_argument(
        "--episodes-dir", default="data/rl_training",
        help="Directory containing *_exit_episodes.parquet files",
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Asset classes to include (e.g. crypto stocks)",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=500_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--output", default="models/dqn_exit_manager",
        help="Output model path (without .zip)",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Use walk-forward training with expanding windows",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.walk_forward:
        train_walk_forward(
            episodes_dir=args.episodes_dir,
            asset_classes=args.classes,
            total_timesteps=args.total_timesteps,
            output_path=args.output,
        )
    else:
        train_single(
            episodes_dir=args.episodes_dir,
            asset_classes=args.classes,
            total_timesteps=args.total_timesteps,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
