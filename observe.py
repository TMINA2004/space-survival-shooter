import argparse

import numpy as np
import pygame
from stable_baselines3 import PPO

from space_shooter.baselines import column_shooter, random_player
from space_shooter.config import Config
from space_shooter.env import SpaceShooterEnv
from space_shooter.rl_player import RLPlayer
from space_shooter.rule_agent import RuleBasedAgent


def parse_args():
    p = argparse.ArgumentParser(description="Watch ONE episode played by an agent.")
    p.add_argument("--agent", choices=["rl", "rule", "column", "random"], default="rl")
    p.add_argument("--model", default="models/ppo_v1/final_model.zip",
                   help="saved RL model (used with --agent rl)")
    p.add_argument("--seed", type=int, default=7, help="seed of the game")
    p.add_argument("--level", type=int, default=4, help="game level (1-4)")
    p.add_argument("--fps", type=int, default=60, help="window speed (0 = as fast as possible)")
    p.add_argument("--stochastic", action="store_true",
                   help="the RL agent samples its actions instead of taking the best one")
    p.add_argument("--hitboxes", action="store_true", help="show the collision circles")
    return p.parse_args()


def make_player(args):
    if args.agent == "rl":
        return RLPlayer(PPO.load(args.model, device="cpu"), deterministic=not args.stochastic)
    return {"rule": RuleBasedAgent(), "column": column_shooter, "random": random_player}[args.agent]


def main():
    args = parse_args()
    player = make_player(args)
    rng = np.random.default_rng(args.seed + 50_000)

    env = SpaceShooterEnv(Config(level=args.level), render_mode="human", render_fps=args.fps)
    env.reset(seed=args.seed)
    env._renderer.show_hitboxes = args.hitboxes

    total_reward, done, paused, quit_requested = 0.0, False, False, False
    while not done and not quit_requested:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_requested = True
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_h:
                    env._renderer.show_hitboxes = not env._renderer.show_hitboxes
        if paused:
            pygame.time.wait(30)
            continue

        obs, reward, terminated, truncated, info = env.step(player(env.game, rng))
        total_reward += reward
        done = terminated or truncated

    stats = env.game.stats
    kills = stats["drifters_killed"] + stats["hunters_killed"]
    accuracy = stats["shots_hit"] / stats["shots_fired"] if stats["shots_fired"] else 0.0
    print(f"Agent: {args.agent} | seed {args.seed} | {env.game.steps} steps | "
          f"total reward {total_reward:.1f}")
    print(f"kills {kills} | accuracy {accuracy:.0%} | lives lost {stats['lives_lost']} | "
          f"dry fires {stats['dry_fires']} | shields {stats['shields_collected']}")

    if not quit_requested:
        print("Episode over. Press ESC or close the window.")
        while True:
            events = pygame.event.get()
            if any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                   for e in events):
                break
            pygame.time.wait(50)
    env.close()


if __name__ == "__main__":
    main()