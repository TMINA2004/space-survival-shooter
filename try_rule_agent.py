import os
import sys

import pandas as pd

from space_shooter.baselines import column_shooter, random_player
from space_shooter.env import SpaceShooterEnv
from space_shooter.evaluation import evaluate_player, summarize
from space_shooter.rule_agent import RuleBasedAgent

# DEVELOPMENT seeds: we use them to debug and tune.
# The final TEST seeds (1000+) are kept untouched for the official comparison.
DEV_SEEDS = range(0, 30)

if len(sys.argv) > 1 and sys.argv[1] == "watch":
    import pygame

    agent = RuleBasedAgent()
    env = SpaceShooterEnv(render_mode="human", render_fps=60)
    obs, _ = env.reset(seed=7)
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                done = True
        obs, reward, terminated, truncated, info = env.step(agent(env.game))
        done = done or terminated or truncated
    print("Episode over after", env.game.steps, "steps. Stats:", env.game.stats)
    env.close()
    sys.exit()

players = {
    "RANDOM": random_player,
    "COLUMN SHOOTER": column_shooter,
    "RULE-BASED": RuleBasedAgent(),
}

frames = []
for name, policy in players.items():
    frames.append(evaluate_player(name, policy, DEV_SEEDS))
    print("done:", name)

df = pd.concat(frames, ignore_index=True)
os.makedirs("results", exist_ok=True)
df.to_csv("results/rule_agent_dev_episodes.csv", index=False)

print("\nDevelopment seeds 0-29 (means):")
print(summarize(df).to_string())