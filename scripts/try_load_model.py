import argparse
import math

import pandas as pd
from stable_baselines3 import PPO

from space_shooter.baselines import column_shooter, random_player
from space_shooter.evaluation import evaluate_player, summarize
from space_shooter.rl_player import RLPlayer
from space_shooter.rule_agent import RuleBasedAgent


def parse_args():
    p = argparse.ArgumentParser(description="Compare a saved PPO model with the reference players.")
    p.add_argument("model", help="path to a saved model, e.g. models/ppo_4m/final_model.zip")
    p.add_argument("--episodes", type=int, default=30, help="number of development games (seeds 0..N-1)")
    return p.parse_args()


def paired_difference(df, column, a="PPO", b="RULE-BASED"):
    """Mean of (a - b) over the SAME games, and its 95% margin of error."""
    table = df.pivot(index="seed", columns="player", values=column)
    diff = table[a] - table[b]
    return diff.mean(), 1.96 * diff.std() / math.sqrt(len(diff))


def main():
    args = parse_args()
    seeds = range(0, args.episodes)
    model = PPO.load(args.model, device="cpu")

    players = {
        "PPO": RLPlayer(model),
        "RULE-BASED": RuleBasedAgent(),
        "COLUMN SHOOTER": column_shooter,
        "RANDOM": random_player,
    }
    frames = []
    for name, player in players.items():
        frames.append(evaluate_player(name, player, seeds))
        print("done:", name, flush=True)
    df = pd.concat(frames, ignore_index=True)

    print(f"\nModel: {args.model} | development seeds 0-{args.episodes - 1} (means)")
    print(summarize(df).to_string())

    print("\nPPO minus RULE-BASED on the same games (mean difference +/- 95% margin):")
    for column in ["total_reward", "kills", "survival_steps"]:
        mean, margin = paired_difference(df, column)
        print(f"  {column:15s}: {mean:+8.1f} +/- {margin:.1f}")
    print("If 'mean +/- margin' does not include 0, the difference is real on these games.")


if __name__ == "__main__":
    main()