import numpy as np
import pandas as pd

from .env import SpaceShooterEnv

SUMMARY_COLUMNS = ["total_reward", "survival_steps", "kills", "accuracy",
                   "shields_collected", "hits_blocked", "dry_fires"]


def play_episode(policy, seed, config=None):
    """Play ONE full episode with policy(game, rng) -> action. Returns a dict of metrics."""
    env = SpaceShooterEnv(config)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 50_000)          # the player's own randomness
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


def evaluate_player(name, policy, seeds, config=None):
    """Play one episode per seed. Returns a table with one row per episode."""
    rows = []
    for seed in seeds:
        row = play_episode(policy, seed, config)
        row["player"] = name
        row["seed"] = seed
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    """One row per player: the means of the main metrics."""
    summary = df.groupby("player")[SUMMARY_COLUMNS].mean().round(2)
    summary["reward_std"] = df.groupby("player")["total_reward"].std().round(2)
    return summary.sort_values("total_reward")