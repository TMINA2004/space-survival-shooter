"""Evaluate the trained RL agents and the reference players on the TEST games.

Examples:
    uv run python test.py
    uv run python test.py --models models/ppo_4m/final_model.zip --episodes 20
    uv run python test.py --max-steps 20000 --out results/test_20k
"""
import argparse
import math
import os

import pandas as pd
from stable_baselines3 import PPO

from space_shooter.baselines import REFERENCE_PLAYERS
from space_shooter.config import Config
from space_shooter.evaluation import evaluate_player, summarize
from space_shooter.rl_player import RLPlayer
from space_shooter.rule_agent import RuleBasedAgent

DEFAULT_MODELS = [
    "models/ppo_4m/final_model.zip",
    "models/ppo_4m_s1/final_model.zip",
    "models/ppo_4m_s2/final_model.zip",
]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate agents on the test games.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="saved PPO models")
    p.add_argument("--episodes", type=int, default=50, help="number of test games")
    p.add_argument("--first-seed", type=int, default=1000, help="first test seed")
    p.add_argument("--max-steps", type=int, default=None, help="episode limit (default: 5000)")
    p.add_argument("--stochastic", action="store_true",
                   help="RL agents sample their actions instead of taking the best one")
    p.add_argument("--out", default="results/test", help="prefix of the CSV files to write")
    return p.parse_args()


def paired_difference(df, column, a, b="RULE-BASED"):
    """Mean of (a - b) over the SAME games, and its 95% margin of error."""
    table = df.pivot(index="seed", columns="player", values=column)
    diff = (table[a] - table[b]).dropna()
    return diff.mean(), 1.96 * diff.std() / math.sqrt(len(diff))


def main():
    args = parse_args()
    for path in args.models:
        if not os.path.exists(path):
            raise SystemExit(f"Model not found: {path}")

    seeds = range(args.first_seed, args.first_seed + args.episodes)
    config = Config(max_episode_steps=args.max_steps) if args.max_steps else Config()

    players = dict(REFERENCE_PLAYERS)                      # STAY, STAY + SHOOT, RANDOM, COLUMN SHOOTER
    players["RULE-BASED"] = RuleBasedAgent()
    ppo_labels = []
    for path in args.models:
        run = os.path.basename(os.path.dirname(path))
        label = f"PPO ({run})"
        players[label] = RLPlayer(PPO.load(path, device="cpu"), deterministic=not args.stochastic)
        ppo_labels.append(label)

    print(f"Test games: seeds {seeds[0]}-{seeds[-1]} ({args.episodes} games), "
          f"episode limit {config.max_episode_steps} steps\n")

    frames = []
    for name, player in players.items():
        frames.append(evaluate_player(name, player, seeds, config))
        print("done:", name, flush=True)
    df = pd.concat(frames, ignore_index=True)

    # One row per player; one extra row with the mean of the PPO seeds
    summary = summarize(df)
    if len(ppo_labels) > 1:
        summary.loc[f"PPO (mean of {len(ppo_labels)} seeds)"] = summary.loc[ppo_labels].mean()
    summary = summary.sort_values("total_reward")

    # Every PPO seed against the rule-based agent, on the same games
    rows = []
    for label in ppo_labels:
        for column in ["total_reward", "kills", "survival_steps", "accuracy"]:
            mean, margin = paired_difference(df, column, label)
            rows.append({"player": label, "metric": column,
                         "mean_difference_vs_rule_based": round(mean, 3),
                         "margin_95": round(margin, 3)})
    paired = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(f"{args.out}_episodes.csv", index=False)          # one row per episode
    summary.to_csv(f"{args.out}_summary.csv")                   # one row per player
    paired.to_csv(f"{args.out}_paired.csv", index=False)        # PPO minus RULE-BASED

    print("\nOFFICIAL RESULTS (means over the test games)")
    print(summary.to_string())
    if len(ppo_labels) > 1:
        spread = summary.loc[ppo_labels, "total_reward"].std()
        print(f"\nStd of the mean reward BETWEEN the PPO seeds: {spread:.1f}")
    print("\nEach PPO seed minus RULE-BASED, same games (mean difference +/- 95% margin):")
    print(paired.to_string(index=False))
    print(f"\nFiles written: {args.out}_episodes.csv, {args.out}_summary.csv, {args.out}_paired.csv")


if __name__ == "__main__":
    main()