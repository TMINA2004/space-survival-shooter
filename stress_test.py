import time

import numpy as np

from space_shooter.config import Config
from space_shooter.env import SpaceShooterEnv

worst = {"enemies": 0, "bullets": 0, "pickups": 0}


def check_step(env, obs, reward):
    """Things that must be true after EVERY step (invariants)."""
    g, cfg = env.game, env.cfg
    size, r = cfg.arena_size, cfg.ship_radius

    assert obs.shape == (25,) and obs.dtype == np.float32, "bad observation shape or type"
    assert np.all(np.isfinite(obs)), f"NaN or inf in the observation: {obs}"
    assert obs.min() >= -1.0 and obs.max() <= 1.0, "observation out of [-1, 1]"
    assert np.isfinite(reward), "bad reward"

    assert r <= g.ship_x <= size - r and r <= g.ship_y <= size - r, "the ship left the arena"
    assert 0 <= g.lives <= cfg.start_lives, f"lives = {g.lives}"
    assert 0 <= g.ammo <= cfg.max_ammo, f"ammo = {g.ammo}"
    assert 0 <= g.shoot_cooldown <= cfg.shoot_cooldown, f"cooldown = {g.shoot_cooldown}"
    assert 0 <= g.invulnerable <= cfg.invulnerable_steps, f"invulnerable = {g.invulnerable}"

    assert len(g.enemies) <= 200, f"too many enemies: {len(g.enemies)}"
    assert len(g.bullets) <= 8, f"too many bullets: {len(g.bullets)}"
    assert len(g.pickups) <= 3, f"too many pickups: {len(g.pickups)}"
    worst["enemies"] = max(worst["enemies"], len(g.enemies))
    worst["bullets"] = max(worst["bullets"], len(g.bullets))
    worst["pickups"] = max(worst["pickups"], len(g.pickups))


def check_episode_end(env, level, terminated, info):
    """Things that must be true when an episode ends (the totals must make sense)."""
    cfg = env.cfg
    s = info["game_stats"]

    assert s["shots_hit"] <= s["shots_fired"], "more hits than shots"
    assert s["drifters_killed"] + s["hunters_killed"] <= s["shots_hit"], "more kills than hits"
    assert s["lives_lost"] + env.game.lives == cfg.start_lives, "lives do not add up"
    assert s["hits_blocked"] <= s["shields_collected"], "more blocks than shields"
    assert info["game_steps"] <= cfg.max_episode_steps, "episode too long"
    if terminated:
        assert env.game.lives == 0, "terminated but still has lives"
    if level < 2:
        assert s["shots_fired"] == 0, "shooting should be off in Level 1"
    if level < 3:
        assert s["dry_fires"] == 0, "no dry fires without limited ammo"
    if level < 4:
        assert s["hunters_killed"] == 0 and s["shields_collected"] == 0, "Level 4 features leaked"


# ---------- Different kinds of weird players ----------
def random_policy():
    return lambda step, rng: int(rng.integers(0, 10))


def sticky_policy():
    """Picks a random action and keeps it for 20 steps (reaches walls and corners)."""
    state = {"action": 0}

    def policy(step, rng):
        if step % 20 == 0:
            state["action"] = int(rng.integers(0, 10))
        return state["action"]

    return policy


def constant_policy(action):
    return lambda step, rng: action


def run_episode(level, policy, seed):
    cfg = Config(level=level, max_episode_steps=3000)
    env = SpaceShooterEnv(cfg)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    steps = 0
    while True:
        action = policy(steps, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        check_step(env, obs, reward)
        if terminated or truncated:
            check_episode_end(env, level, terminated, info)
            return steps


start = time.time()
total_steps = total_episodes = 0

for level in (1, 2, 3, 4):
    jobs = []
    jobs += [("random", random_policy(), s) for s in range(20)]
    jobs += [("sticky random", sticky_policy(), 100 + s) for s in range(10)]
    jobs += [(f"always action {a}", constant_policy(a), 200 + a) for a in range(10)]

    for name, policy, seed in jobs:
        try:
            total_steps += run_episode(level, policy, seed)
        except AssertionError as error:
            print(f"FAILED at level {level}, player '{name}', seed {seed}: {error}")
            raise
        total_episodes += 1
    print(f"Level {level}: {len(jobs)} episodes OK")

seconds = time.time() - start
print(f"\nAll good: {total_episodes} episodes, {total_steps:,} steps checked in {seconds:.0f} s")
print("Most objects alive at once:", worst)