import numpy as np

from space_shooter.config import Config
from space_shooter.game import (
    Game, Enemy, Pickup, DRIFTER, HUNTER, STAY, make_action,
)
import math

NO_SPAWN = dict(spawn_start_interval=10**9, spawn_min_interval=10**9)


def distance(game, enemy):
    return math.hypot(enemy.x - game.ship_x, enemy.y - game.ship_y)


# ---------- Test 1: a Hunter walks toward the ship at speed 1.5 ----------
game = Game(Config(level=4, **NO_SPAWN), seed=0)
hunter = Enemy(x=100, y=100, kind=HUNTER, hp=2)
game.enemies.append(hunter)
before = distance(game, hunter)
game.step(STAY)
after = distance(game, hunter)
assert abs((before - after) - 1.5) < 1e-9, "Hunter speed or direction is wrong"
print("Test 1 OK: Hunter gets 1.5 closer each step")

# ---------- Test 2: a Hunter that is not stopped hits the ship ----------
game = Game(Config(level=4, **NO_SPAWN), seed=0)
game.enemies.append(Enemy(x=100, y=100, kind=HUNTER, hp=2))
for _ in range(400):
    game.step(STAY)
assert game.lives == 2 and len(game.enemies) == 0
print("Test 2 OK: an unstopped Hunter costs one life")

# ---------- Test 3: a Hunter needs exactly 2 hits ----------
game = Game(Config(level=4, **NO_SPAWN), seed=0)
game.enemies.append(Enemy(x=game.ship_x, y=game.ship_y - 300, kind=HUNTER, hp=2))
for _ in range(80):
    game.step(make_action(STAY, True))
assert game.stats["shots_hit"] == 2 and game.stats["hunters_killed"] == 1
assert game.lives == 3
print("Test 3 OK: Hunter dies after exactly 2 hits")

# ---------- Test 4: Hunters only exist in Level 4 ----------
for level in (3, 4):
    game = Game(Config(level=level, start_lives=10**6), seed=0)
    seen_hunter = False
    for _ in range(3000):
        game.step(STAY)
        if any(e.kind == HUNTER for e in game.enemies):
            seen_hunter = True
    assert seen_hunter == (level == 4), f"Hunter rule wrong at level {level}"
print("Test 4 OK: Hunters appear only in Level 4")

# ---------- Test 5: a shield pickup falls and is collected ----------
game = Game(Config(level=4, **NO_SPAWN), seed=0)
game.pickups.append(Pickup(x=game.ship_x, y=game.ship_y - 100))
for _ in range(60):
    game.step(STAY)
assert game.shield and game.stats["shields_collected"] == 1 and len(game.pickups) == 0
print("Test 5 OK: shield pickup collected")

# ---------- Test 6: a shield absorbs one hit ----------
game = Game(Config(level=4, **NO_SPAWN), seed=0)
game.shield = True
game.enemies.append(Enemy(x=game.ship_x, y=game.ship_y, kind=DRIFTER))
events = game.step(STAY)
assert game.lives == 3 and not game.shield
assert events["hits_blocked"] == 1 and game.invulnerable > 0
print("Test 6 OK: shield took the hit, no life lost")

# ---------- Test 7: pickups exist only in Level 4 ----------
for level in (3, 4):
    game = Game(Config(level=level, **NO_SPAWN), seed=0)
    appeared = False
    for _ in range(600):
        game.step(STAY)
        if game.pickups:
            appeared = True
    assert appeared == (level == 4), f"pickup rule wrong at level {level}"
print("Test 7 OK: shield pickups appear only in Level 4")

# ---------- Test 8: same seed + same actions = same game (Level 4) ----------
actions = np.random.default_rng(99).integers(0, 10, 1000)
g1, g2 = Game(seed=7), Game(seed=7)
for a in actions:
    g1.step(a)
    g2.step(a)
state1 = (g1.lives, g1.stats, [(e.x, e.y, e.kind, e.hp) for e in g1.enemies],
          [(p.x, p.y) for p in g1.pickups])
state2 = (g2.lives, g2.stats, [(e.x, e.y, e.kind, e.hp) for e in g2.enemies],
          [(p.x, p.y) for p in g2.pickups])
assert state1 == state2
print("Test 8 OK: Level 4 is still reproducible with seeds")


# ---------- Baselines for the full game (Level 4) ----------
def play(policy, seed):
    g = Game(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    while not g.game_over and g.steps < 20_000:
        g.step(policy(g, rng))
    return g


def column_shooter(g, rng):
    threat = any(abs(e.x - g.ship_x) < 25 and e.y < g.ship_y for e in g.enemies)
    return make_action(STAY, threat)


policies = {
    "STAY + SHOOT": lambda g, rng: make_action(STAY, True),
    "RANDOM": lambda g, rng: int(rng.integers(0, 10)),
    "COLUMN SHOOTER": column_shooter,
}

print("\nFull game (Level 4), 20 games each:")
for name, policy in policies.items():
    games = [play(policy, s) for s in range(20)]
    survival = np.mean([g.steps for g in games])
    drifters = np.mean([g.stats["drifters_killed"] for g in games])
    hunters = np.mean([g.stats["hunters_killed"] for g in games])
    shields = np.mean([g.stats["shields_collected"] for g in games])
    blocked = np.mean([g.stats["hits_blocked"] for g in games])
    print(f"{name:15s}: survival={survival:7.0f}  drifters_killed={drifters:5.1f}  "
          f"hunters_killed={hunters:5.1f}  shields={shields:4.1f}  blocked={blocked:4.1f}")