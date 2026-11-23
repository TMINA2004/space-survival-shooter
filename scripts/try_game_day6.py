import numpy as np

from space_shooter.config import Config
from space_shooter.game import Game, STAY, make_action

NO_SPAWN = dict(spawn_start_interval=10**9, spawn_min_interval=10**9)

# ---------- Test 1: start with 20 ammo ----------
game = Game(Config(**NO_SPAWN), seed=0)
assert game.ammo == 20
print("Test 1 OK: starts with 20 ammo")

# ---------- Test 2: a shot costs 1 ammo, and it comes back after 20 steps ----------
game = Game(Config(**NO_SPAWN), seed=0)
game.step(make_action(STAY, True))
assert game.ammo == 19
for _ in range(19):
    game.step(STAY)
assert game.ammo == 19, "recharge came too early"
game.step(STAY)
assert game.ammo == 20, "recharge is wrong"
print("Test 2 OK: 1 shot costs 1 ammo, recharged 20 steps later")

# ---------- Test 3: dry fire ----------
game = Game(Config(**NO_SPAWN), seed=0)
game.ammo = 0
game.step(make_action(STAY, True))
assert len(game.bullets) == 0
assert game.stats["dry_fires"] == 1 and game.stats["shots_fired"] == 0
print("Test 3 OK: empty magazine = dry fire, no bullet")

# ---------- Test 4: ammo never goes above the maximum ----------
game = Game(Config(**NO_SPAWN), seed=0)
for _ in range(100):
    game.step(STAY)
assert game.ammo == 20
print("Test 4 OK: ammo never above 20")

# ---------- Test 5: holding fire for 400 steps ----------
game = Game(Config(**NO_SPAWN), seed=0)
for _ in range(400):
    game.step(make_action(STAY, True))
fired = game.stats["shots_fired"]
assert 35 <= fired <= 39, f"unexpected number of shots: {fired}"
assert game.stats["dry_fires"] > 0
print(f"Test 5 OK: 400 steps of holding fire = {fired} shots (20 in the magazine + recharges)")

# ---------- Test 6: Level 2 has unlimited ammo ----------
game = Game(Config(level=2, **NO_SPAWN), seed=0)
for _ in range(400):
    game.step(make_action(STAY, True))
assert game.stats["shots_fired"] == 50 and game.stats["dry_fires"] == 0 and game.ammo == 20
print("Test 6 OK: Level 2 = unlimited ammo (50 shots, limited only by the cooldown)")

# ---------- Baselines with limited ammo ----------
def play(policy, seed):
    g = Game(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    while not g.game_over and g.steps < 20_000:
        g.step(policy(g, rng))
    return g


def column_shooter(g, rng):
    """Stays still and shoots ONLY when an enemy is above it, in its column."""
    threat = any(abs(e.x - g.ship_x) < 25 and e.y < g.ship_y for e in g.enemies)
    return make_action(STAY, threat)


policies = {
    "STAY + SHOOT (wastes ammo)": lambda g, rng: make_action(STAY, True),
    "RANDOM": lambda g, rng: int(rng.integers(0, 10)),
    "COLUMN SHOOTER": column_shooter,
}

print("\n(20 games each. Survival = 20000 means it hit our test cap.)")
for name, policy in policies.items():
    games = [play(policy, s) for s in range(20)]
    steps = np.mean([g.steps for g in games])
    kills = np.mean([g.stats["drifters_killed"] for g in games])
    fired = sum(g.stats["shots_fired"] for g in games)
    hits = sum(g.stats["shots_hit"] for g in games)
    dry = np.mean([g.stats["dry_fires"] for g in games])
    accuracy = 100 * hits / max(1, fired)
    print(f"{name:27s}: survival={steps:7.0f}  kills={kills:6.1f}  "
          f"accuracy={accuracy:5.1f}%  dry_fires={dry:7.1f}")