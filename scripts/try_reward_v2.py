from space_shooter.config import Config
from space_shooter.env import SpaceShooterEnv
from space_shooter.game import Pickup, STAY

NO_SPAWN = dict(spawn_start_interval=10**9, spawn_min_interval=10**9)
env = SpaceShooterEnv(Config(level=4, **NO_SPAWN))
env.reset(seed=0)
g = env.game

g.pickups.append(Pickup(x=g.ship_x, y=g.ship_y))
obs, reward, terminated, truncated, info = env.step(STAY)
assert g.shield and abs(reward - 0.51) < 1e-9
print("OK: the first ring pays +0.5")

g.pickups.append(Pickup(x=g.ship_x, y=g.ship_y))
obs, reward, terminated, truncated, info = env.step(STAY)
assert g.shield and abs(reward - 0.01) < 1e-9
print("OK: a ring collected while shielded pays nothing")