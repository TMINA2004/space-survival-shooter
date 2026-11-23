
import argparse
import json
import os
import platform
import sys
import time
from importlib.metadata import version

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env

from space_shooter.callbacks import (
    EpisodeStatsCallback,
    PeriodicEvalCallback,
)
from space_shooter.config import Config
from space_shooter.env import SpaceShooterEnv
from space_shooter.network import SpaceFeaturesExtractor


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a PPO agent on Space Survival Shooter."
    )

    p.add_argument("--name", default="ppo", help="Run name")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--level", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--features-dim", type=int, default=128)

    p.add_argument(
        "--network",
        choices=["custom", "mlp"],
        default="custom",
        help="custom = structured network, mlp = plain SB3 MLP",
    )

    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=250_000)

    return p.parse_args()


def main():
    args = parse_args()

    model_dir = os.path.join("models", args.name)
    os.makedirs(
        os.path.join(model_dir, "checkpoints"),
        exist_ok=True,
    )

    config = Config(level=args.level)

    # Training seeds start at 10,000.
    # They do not overlap validation, development or test seeds.
    env = make_vec_env(
        SpaceShooterEnv,
        n_envs=args.n_envs,
        seed=10_000 + args.seed * 100,
        env_kwargs=dict(config=config),
    )

    # Choose the network configuration BEFORE creating the PPO model.
    if args.network == "custom":
        policy_kwargs = dict(
            features_extractor_class=SpaceFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=args.features_dim,
            ),
            net_arch=dict(
                pi=[64],
                vf=[64],
            ),
        )
    else:
        # Plain MLP baseline.
        policy_kwargs = dict(
            net_arch=dict(
                pi=[128, 128],
                vf=[128, 128],
            ),
        )

    # Create the PPO model.
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device="cpu",
        verbose=0,
        tensorboard_log="runs",
    )

    callbacks = CallbackList([
        EpisodeStatsCallback(),

        PeriodicEvalCallback(
            model_dir,
            args.eval_every,
            range(2000, 2000 + args.eval_episodes),
            config,
        ),

        CheckpointCallback(
            save_freq=max(
                args.checkpoint_every // args.n_envs,
                1,
            ),
            save_path=os.path.join(
                model_dir,
                "checkpoints",
            ),
            name_prefix="ckpt",
        ),
    ])

    # Information required for the experiment report.
    run_info = {
        "settings": vars(args),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            package: version(package)
            for package in [
                "torch",
                "stable-baselines3",
                "gymnasium",
                "numpy",
                "pandas",
                "pygame",
            ]
        },
        "network_parameters": sum(
            p.numel()
            for p in model.policy.parameters()
        ),
    }

    print(
        f"Run '{args.name}': {args.steps:,} steps, "
        f"{args.n_envs} parallel games, level {args.level}"
    )
    print("TensorBoard: uv run tensorboard --logdir runs")
    print(
        f"Network: {args.network} "
        f"({run_info['network_parameters']:,} parameters)"
    )

    start = time.time()

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            tb_log_name=args.name,
        )

    except KeyboardInterrupt:
        print("\nInterrupted: saving the model reached so far.")

    finally:
        seconds = time.time() - start

        # Save the final model.
        model.save(
            os.path.join(model_dir, "final_model")
        )

        run_info["training_seconds"] = round(seconds)
        run_info["steps_per_second"] = (
            round(model.num_timesteps / seconds)
            if seconds > 0
            else 0
        )
        run_info["steps_done"] = model.num_timesteps

        with open(
            os.path.join(model_dir, "run_info.json"),
            "w",
        ) as f:
            json.dump(run_info, f, indent=2)

        print(
            f"\nFinished: {model.num_timesteps:,} steps "
            f"in {seconds / 60:.1f} min "
            f"({run_info['steps_per_second']:,} steps/s)"
        )

        print(f"Final model: {model_dir}/final_model.zip")
        print(f"Best model: {model_dir}/best_model.zip")


if __name__ == "__main__":
    main()
