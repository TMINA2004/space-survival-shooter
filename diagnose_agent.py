import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from space_shooter.baselines import column_shooter
from space_shooter.env import SpaceShooterEnv
from space_shooter.rl_player import RLPlayer
from space_shooter.rule_agent import RuleBasedAgent

SEEDS = range(0, 5)
MOVES = ["stay", "up", "down", "left", "right"]


def profile(name, player):
    moves = np.zeros(5)
    steps = shoot_pressed = dry = side = 0
    ys, dxs = [], []
    fired = hit = kills = 0

    for seed in SEEDS:
        env = SpaceShooterEnv()
        env.reset(seed=seed)
        game = env.game
        rng = np.random.default_rng(seed + 50_000)
        done = False
        while not done:
            action = player(game, rng)
            moves[action % 5] += 1
            if action >= 5:                          # the fire button is pressed
                shoot_pressed += 1
                dry += int(game.ammo == 0)           # ... with an empty magazine
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            ys.append(game.ship_y)
            dxs.append(abs(game.ship_x - 300))
            side += int(game.ship_x < 45 or game.ship_x > 555)    # hugging a side wall
        s = info["game_stats"]
        fired += s["shots_fired"]
        hit += s["shots_hit"]
        kills += s["drifters_killed"] + s["hunters_killed"]

    row = {"player": name,
           "steps/game": round(steps / len(SEEDS)),
           "fire pressed %": round(100 * shoot_pressed / steps, 1),
           "fire on empty %": round(100 * dry / steps, 1),
           "accuracy %": round(100 * hit / max(fired, 1), 1),
           "kills/game": round(kills / len(SEEDS), 1),
           "mean y": round(float(np.mean(ys))),
           "mean |x-300|": round(float(np.mean(dxs))),
           "at side walls %": round(100 * side / steps, 1)}
    for i, move in enumerate(MOVES):
        row[f"{move} %"] = round(100 * moves[i] / steps, 1)
    return row


path = sys.argv[1] if len(sys.argv) > 1 else "models/ppo_v1/final_model.zip"
players = {
    "PPO": RLPlayer(PPO.load(path, device="cpu")),
    "RULE-BASED": RuleBasedAgent(),
    "COLUMN SHOOTER": column_shooter,
}
rows = []
for name, player in players.items():
    rows.append(profile(name, player))
    print("done:", name)

print(f"\nBehavior profile of {path} (5 games each)")
print(pd.DataFrame(rows).set_index("player").T.to_string()) 