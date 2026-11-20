import sys
import numpy as np
from stable_baselines3 import PPO

from space_shooter.config import Config
from space_shooter.env import SpaceShooterEnv
from space_shooter.rl_player import RLPlayer
from space_shooter.evaluation import evaluate_player, summarize
from space_shooter.baselines import REFERENCE_PLAYERS


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run python try_load_model.py <model_path>")
        return

    model_path = sys.argv[1]
    model = PPO.load(model_path)
    player = RLPlayer(model, deterministic=True)

    seeds = range(30)
    config = Config(level=4)

    results = []

    for name, policy in REFERENCE_PLAYERS.items():
        results.append(evaluate_player(name, policy, seeds, config))

    results.append(evaluate_player("PPO", player, seeds, config))

    import pandas as pd
    df = pd.concat(results, ignore_index=True)
    print(summarize(df))


if __name__ == "__main__":
    main()