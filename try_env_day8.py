import sys
import time

import numpy as np
from stable_baselines3.common.env_checker import check_env

from space_shooter.config import Config
from space_shooter.env import SpaceShooterEnv
from space_shooter.game import Enemy, Pickup, DRIFTER, HUNTER, STAY

NO_SPAWN = dict(spawn_start_interval=10**9, spawn_min_interval=10**9)

# ---------- Test 1: the official Gymnasium / SB3 checker ----------
env = SpaceShooterEnv()
check_env(env, warn=True)
print("Test 1 OK: passes the Stable-Baselines3 environment checker")

# ---------- Test 2: spaces ----------
assert env.observation_space.shape == (25,) and env.action_space.n == 10
print("Test 2 OK: 25 observations, 10 actions")

# ---------- Test 3: the first observation ----------
obs, info = env.reset(seed=0)
assert obs.shape == (25,) and obs.dtype == np.float32
assert obs.min() >= -1 and obs.max() <= 1
assert abs(obs[0] - 0.5) < 1e-6 and abs(obs[1] - 0.9) < 1e-6
assert obs[2] == 1.0 and obs[3] == 1.0 and obs[4] == 0.0
assert np.all(obs[6:] == 0)
print("Test 3 OK: first observation is correct")

# ---------- Test 4: enemies in the state, nearest first ----------
env = SpaceShooterEnv(Config(level=4, **NO_SPAWN))
env.reset(seed=0)
g = env.game
far = Enemy(x=g.ship_x - 200, y=g.ship_y - 300, kind=DRIFTER)
near = Enemy(x=g.ship_x + 60, y=g.ship_y - 120, kind=HUNTER, hp=2)
g.enemies.extend([far, near])
obs = env._observation()
assert abs(obs[6] - 0.1) < 1e-6 and abs(obs[7] + 0.2) < 1e-6
assert obs[8] == 1.0 and obs[9] == 1.0                      # slot 1 = the nearby Hunter
assert abs(obs[10] + 200 / 600) < 1e-6 and obs[12] == 0.0 and obs[13] == 1.0   # slot 2 = far Drifter
assert np.all(obs[14:22] == 0)                              # empty slots are zeros
print("Test 4 OK: enemies are sorted by distance and scaled")

# ---------- Test 5: the shield pickup in the state ----------
g.pickups.append(Pickup(x=g.ship_x, y=g.ship_y - 60))
obs = env._observation()
assert obs[22] == 0 and abs(obs[23] + 0.1) < 1e-6 and obs[24] == 1.0
print("Test 5 OK: pickup in the state")

# ---------- Test 6: the reward function ----------
env = SpaceShooterEnv()


def events(**changes):
    base = {"lives_lost": 0, "shots_fired": 0, "shots_hit": 0, "dry_fires": 0,
            "drifters_killed": 0, "hunters_killed": 0, "shields_collected": 0,
            "hits_blocked": 0, "game_over": False}
    base.update(changes)
    return base


assert abs(env._reward(events()) - 0.01) < 1e-9
assert abs(env._reward(events(drifters_killed=1)) - 1.01) < 1e-9
assert abs(env._reward(events(hunters_killed=1)) - 2.01) < 1e-9
assert abs(env._reward(events(lives_lost=1)) - (-4.99)) < 1e-9
assert abs(env._reward(events(shields_collected=1)) - 0.51) < 1e-9
print("Test 6 OK: reward values are correct")

# ---------- Test 7: terminated ----------
env = SpaceShooterEnv(Config(**NO_SPAWN))
env.reset(seed=0)
env.game.lives = 1
env.game.enemies.append(Enemy(x=env.game.ship_x, y=env.game.ship_y))
obs, reward, terminated, truncated, info = env.step(STAY)
assert terminated and not truncated and reward < 0 and "game_stats" in info
print("Test 7 OK: losing the last life = terminated, with the episode stats in info")

# ---------- Test 8: truncated ----------
env = SpaceShooterEnv(Config(max_episode_steps=50, **NO_SPAWN))
env.reset(seed=0)
for i in range(50):
    obs, reward, terminated, truncated, info = env.step(STAY)
    assert not terminated
    assert truncated == (i == 49)
assert info["game_steps"] == 50
print("Test 8 OK: episode is truncated at the step limit")

# ---------- Test 9: same seed + same actions = same observations ----------
actions = np.random.default_rng(1).integers(0, 10, 300)
runs = []
for _ in range(2):
    env = SpaceShooterEnv()
    obs, _ = env.reset(seed=3)
    seq = [obs]
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(int(a))
        seq.append(obs)
        if terminated or truncated:
            break
    runs.append(np.array(seq))
assert runs[0].shape == runs[1].shape and np.array_equal(runs[0], runs[1])
print("Test 9 OK: the environment is reproducible")

# ---------- Headless check (information) ----------
print("pygame loaded in headless mode?", "pygame" in sys.modules, "(should be False)")

# ---------- The first RL baseline: a random agent through the real environment ----------
env = SpaceShooterEnv()
env.action_space.seed(0)
totals, lengths = [], []
start = time.time()
total_steps = 0
for episode in range(20):
    obs, _ = env.reset(seed=episode)
    done, total = False, 0.0
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        total_steps += 1
        done = terminated or truncated
    totals.append(total)
    lengths.append(info["game_steps"])

print(f"\nRandom agent, 20 episodes: total reward = {np.mean(totals):.2f} "
      f"(+/- {np.std(totals):.2f}), episode length = {np.mean(lengths):.0f} steps")
print(f"Environment speed: {total_steps / (time.time() - start):,.0f} steps per second")