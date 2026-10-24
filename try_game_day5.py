import time

import numpy as np

from space_shooter.config import Config
from space_shooter.game import (
    Game, Enemy, DRIFTER, STAY, UP, DOWN, LEFT, RIGHT, make_action, split_action,
)

# A config where NO enemy spawns by itself, so tests are exact
NO_SPAWN = Config(spawn_start_interval=10**9, spawn_min_interval=10**9)

# ---------- Test 1: action encoding ----------
assert make_action(LEFT, True) == 8
assert split_action(8) == (LEFT, True)
assert split_action(2) == (DOWN, False)
print("Test 1 OK: action encoding works")

# ---------- Test 2: shooting creates a bullet at the ship's tip ----------
game = Game(NO_SPAWN, seed=0)
game.step(make_action(STAY, True))
assert len(game.bullets) == 1 and game.stats["shots_fired"] == 1
assert game.bullets[0].x == 300 and game.bullets[0].y == 540 - 15 - 12
print("Test 2 OK: bullet created and moved up")

# ---------- Test 3: cooldown (1 shot per 8 steps) ----------
game = Game(NO_SPAWN, seed=0)
for _ in range(8):
    game.step(make_action(STAY, True))
assert game.stats["shots_fired"] == 1, "cooldown is wrong"
game.step(make_action(STAY, True))
assert game.stats["shots_fired"] == 2, "second shot should be allowed at step 9"
print("Test 3 OK: cooldown works")

# ---------- Test 4: move and shoot in the same step ----------
game = Game(NO_SPAWN, seed=0)
game.step(make_action(LEFT, True))
assert game.ship_x == 294 and len(game.bullets) == 1 and game.bullets[0].x == 294
print("Test 4 OK: move + shoot together")

# ---------- Test 5: a bullet kills a Drifter ----------
game = Game(NO_SPAWN, seed=0)
game.enemies.append(Enemy(x=game.ship_x, y=game.ship_y - 100, kind=DRIFTER, hp=1))
game.step(make_action(STAY, True))
for _ in range(20):
    game.step(STAY)
assert game.stats["drifters_killed"] == 1 and game.stats["shots_hit"] == 1
assert len(game.enemies) == 0 and game.lives == 3
print("Test 5 OK: bullet destroys a Drifter")

# ---------- Test 6: an enemy with 2 HP needs 2 hits ----------
game = Game(NO_SPAWN, seed=0)
game.enemies.append(Enemy(x=game.ship_x, y=game.ship_y - 300, kind=DRIFTER, hp=2))
for _ in range(60):
    game.step(make_action(STAY, True))
assert game.stats["shots_hit"] == 2 and game.stats["drifters_killed"] == 1
print("Test 6 OK: 2 HP enemy dies after exactly 2 hits")

# ---------- Test 7: bullets leave the arena ----------
game = Game(NO_SPAWN, seed=0)
game.step(make_action(STAY, True))
for _ in range(100):
    game.step(STAY)
assert len(game.bullets) == 0
print("Test 7 OK: bullets that miss are removed")

# ---------- Test 8: Level 1 has no shooting ----------
game = Game(Config(level=1, spawn_start_interval=10**9, spawn_min_interval=10**9), seed=0)
game.step(make_action(STAY, True))
assert len(game.bullets) == 0 and game.stats["shots_fired"] == 0
print("Test 8 OK: Level 1 ignores shooting")

# ---------- Test 9: invalid actions are rejected ----------
try:
    Game(seed=0).step(10)
    raise SystemExit("Test 9 FAILED: action 10 should be rejected")
except ValueError:
    print("Test 9 OK: invalid action rejected")

# ---------- Baselines ----------
def play(policy, seed):
    g = Game(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    while not g.game_over and g.steps < 20_000:
        g.step(policy(rng))
    return g.steps, g.stats

policies = {
    "STAY (no shooting)": lambda rng: STAY,
    "STAY + SHOOT": lambda rng: make_action(STAY, True),
    "RANDOM (10 actions)": lambda rng: int(rng.integers(0, 10)),
}
print()
for name, policy in policies.items():
    results = [play(policy, s) for s in range(20)]
    steps = np.mean([r[0] for r in results])
    kills = np.mean([r[1]["drifters_killed"] for r in results])
    print(f"{name:22s}: survival = {steps:7.0f} steps, kills = {kills:5.1f}")

# ---------- Speed ----------
game = Game(Config(start_lives=10_000_000), seed=0)
start = time.time()
for _ in range(200_000):
    game.step(make_action(STAY, True))
print(f"\nSpeed: {200_000 / (time.time() - start):,.0f} steps per second")