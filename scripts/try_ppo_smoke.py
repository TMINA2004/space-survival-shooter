import time
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from space_shooter.baselines import random_player
from space_shooter.callbacks import EpisodeStatsCallback
from space_shooter.env import SpaceShooterEnv
from space_shooter.evaluation import evaluate_player, summarize
from space_shooter.network import SpaceFeaturesExtractor
from space_shooter.rl_player import RLPlayer


# ==========================================
# TRAINING CONFIGURATION
# ==========================================

N_ENVS = 8
TOTAL_STEPS = 100_000

DEV_SEEDS = range(0, 10)


if __name__ == "__main__":

    # ==========================================
    # CREATE VECTOR ENVIRONMENT
    # ==========================================

    # 8 games played simultaneously
    env = make_vec_env(
        SpaceShooterEnv,
        n_envs=N_ENVS,
        seed=0
    )

    # ==========================================
    # CREATE PPO MODEL
    # ==========================================

    model = PPO(
        "MlpPolicy",
        env,

        policy_kwargs=dict(

            # Our custom feature extractor
            features_extractor_class=SpaceFeaturesExtractor,

            features_extractor_kwargs=dict(
                features_dim=128
            ),

            # Actor and critic networks
            net_arch=dict(
                pi=[64],
                vf=[64]
            ),
        ),

        learning_rate=3e-4,

        # Steps per environment before each update
        n_steps=1024,

        batch_size=256,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.2,
        ent_coef=0.01,

        seed=0,
        device="cpu",
        verbose=0,

        tensorboard_log="runs/day12",
    )

    # ==========================================
    # COUNT PARAMETERS
    # ==========================================

    total_params = sum(
        p.numel()
        for p in model.policy.parameters()
    )

    print(
        "Total parameters (network + heads):",
        total_params
    )

    # ==========================================
    # TRAIN THE AGENT
    # ==========================================

    start = time.time()

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=EpisodeStatsCallback(),
        tb_log_name="ppo_smoke"
    )

    seconds = time.time() - start

    steps_per_second = TOTAL_STEPS / seconds

    print(
        f"\nTraining: {TOTAL_STEPS:,} steps "
        f"in {seconds:.0f} s "
        f"= {steps_per_second:,.0f} steps/s"
    )

    print(
        f"Estimated time for 1,000,000 steps: "
        f"{1_000_000 / steps_per_second / 60:.0f} minutes"
    )

    # ==========================================
    # EVALUATE THE TRAINED AGENT
    # ==========================================

    # Play 10 development games (seeds 0-9)
    # Compare PPO against RANDOM

    frames = [

        evaluate_player(
            "PPO (100k steps)",
            RLPlayer(model),
            DEV_SEEDS
        ),

        evaluate_player(
            "RANDOM",
            random_player,
            DEV_SEEDS
        ),
    ]

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    print("\nDevelopment seeds 0-9 (means):")

    results = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        summarize(results).to_string()
    )

    # ==========================================
    # SAVE TRAINED MODEL
    # ==========================================

    model.save("space_shooter_ppo")

    print("\nModel saved successfully!")

    env.close()