import time

import numpy as np

from space_shooter.config import Config
from space_shooter.game import (
    Game, Enemy, DRIFTER, STAY, UP, DOWN, LEFT, RIGHT
)

# Test 1: the first enemy appears after 40 steps
game = Game(Config(level=3), seed=0)
for _ in range(40):
    game.step(STAY)

assert len(game.enemies) == 1, "first spawn is wrong"
print("Test 1 OK: first enemy spawned at step 40")

# Test 2: enemies fall by 3 per step
y_before = game.enemies[0].y
game.step(STAY)

assert game.enemies[0].y == y_before + 3, "falling speed is wrong"
print("Test 2 OK: Drifter falls 3 per step")

# Test 3: a collision removes a life and the enemy
game = Game(seed=0)
game.enemies.append(
    Enemy(x=game.ship_x, y=game.ship_y, kind=DRIFTER)
)
events = game.step(STAY)

assert game.lives == 2 and events["lives_lost"] == 1, "collision is wrong"
assert len(game.enemies) == 0, "enemy should be destroyed"
assert game.invulnerable > 0, "ship should be protected"
print("Test 3 OK: collision costs 1 life, enemy destroyed")

# Test 4: while protected, no life is lost
game.enemies.append(
    Enemy(x=game.ship_x, y=game.ship_y, kind=DRIFTER)
)
game.step(STAY)

assert game.lives == 2 and len(game.enemies) == 0, "invulnerability is wrong"
print("Test 4 OK: protected ship loses nothing, enemy still destroyed")

# Test 5: game over at 0 lives
game = Game(seed=0)
game.lives = 1
game.enemies.append(
    Enemy(x=game.ship_x, y=game.ship_y, kind=DRIFTER)
)
events = game.step(STAY)

assert game.game_over and events["game_over"], "game over is wrong"
steps_at_end = game.steps
game.step(STAY)

assert game.steps == steps_at_end, "game must freeze after game over"
print("Test 5 OK: game over works and the game freezes")

# Test 6: the difficulty ramp
game = Game()
results = []

for s in [0, 2500, 5000, 10000]:
    game.steps = s
    results.append(game.spawn_interval())

assert results == [40, 27, 15, 15], f"ramp is wrong: {results}"
print("Test 6 OK: spawn interval =", results)

# Test 7: same seed = same game
g1, g2 = Game(seed=5), Game(seed=5)

for _ in range(500):
    g1.step(STAY)
    g2.step(STAY)

pos1 = [(e.x, e.y) for e in g1.enemies]
pos2 = [(e.x, e.y) for e in g2.enemies]

assert pos1 == pos2, "seeds are not reproducible"
print("Test 7 OK: same seed gives the same enemies")

# Test 8: off-screen enemies are removed
game = Game(Config(start_lives=10_000), seed=1)
most = 0

for _ in range(5000):
    game.step(STAY)
    most = max(most, len(game.enemies))

assert most < 30, f"too many enemies alive: {most}"
print("Test 8 OK: at most", most, "enemies alive at the same time")

# Baselines: how long do simple players survive?
def survival(policy, seed):
    g = Game(seed=seed)
    rng = np.random.default_rng(seed + 1000)

    while not g.game_over and g.steps < 100_000:
        g.step(policy(rng))

    return g.steps


stay_scores = [survival(lambda rng: STAY, s) for s in range(20)]
random_scores = [
    survival(lambda rng: int(rng.integers(0, 5)), s)
    for s in range(20)
]

print(f"\nBaseline STAY   : average survival = {np.mean(stay_scores):.0f} steps")
print(f"Baseline RANDOM : average survival = {np.mean(random_scores):.0f} steps")

# Speed test
game = Game(Config(start_lives=10_000_000), seed=0)
start = time.time()

for _ in range(200_000):
    game.step(STAY)

print(f"Speed: {200_000 / (time.time() - start):,.0f} steps per second")