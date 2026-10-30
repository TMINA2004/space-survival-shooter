import os

import pandas as pd

from space_shooter.baselines import REFERENCE_PLAYERS
from space_shooter.evaluation import evaluate_player, summarize
from space_shooter.rule_agent import RuleBasedAgent

# TEST seeds: kept untouched until now. Training uses other seeds.
EVAL_SEEDS = range(1000, 1050)

players = dict(REFERENCE_PLAYERS)
players["RULE-BASED"] = RuleBasedAgent()

frames = []
for name, policy in players.items():
    frames.append(evaluate_player(name, policy, EVAL_SEEDS))
    print("done:", name)

df = pd.concat(frames, ignore_index=True)
os.makedirs("results", exist_ok=True)
df.to_csv("results/reference_episodes.csv", index=False)       # one row per episode

summary = summarize(df)
summary.to_csv("results/reference_summary.csv")                # one row per player

print("\nOfficial reference table, test seeds 1000-1049 (means):")
print(summary.to_string())