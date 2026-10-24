import os

import numpy as np
import pandas as pd

from space_shooter.baselines import REFERENCE_PLAYERS
from space_shooter.env import SpaceShooterEnv

# Evaluation seeds. Training will use OTHER seeds, so we never test on the games we trained on.
EVAL_SEEDS = range(1000, 1050)


def play_episode(policy, seed):
    env = SpaceShooterEnv()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 50_000)      # the player's own randomness
    total_reward, done = 0.0, False
    while not done:
        action = policy(env.game, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    row = dict(info["game_stats"])
    row["total_reward"] = total_reward
    row["survival_steps"] = info["game_steps"]
    row["kills"] = row["drifters_killed"] + row["hunters_killed"]
    row["accuracy"] = row["shots_hit"] / row["shots_fired"] if row["shots_fired"] > 0 else np.nan
    return row


rows = []
for name, policy in REFERENCE_PLAYERS.items():
    for seed in EVAL_SEEDS:
        row = play_episode(policy, seed)
        row["player"] = name
        row["seed"] = seed
        rows.append(row)
    print("done:", name)

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/reference_episodes.csv", index=False)        # one row per episode

columns = ["total_reward", "survival_steps", "kills", "accuracy",
           "shields_collected", "dry_fires"]
summary = df.groupby("player")[columns].mean().round(2)
summary["reward_std"] = df.groupby("player")["total_reward"].std().round(2)
summary = summary.sort_values("survival_steps")
summary.to_csv("results/reference_summary.csv")                  # one row per player

print("\nReference players, 50 episodes each (means):")
print(summary.to_string())