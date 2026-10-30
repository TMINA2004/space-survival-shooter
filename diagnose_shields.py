from space_shooter.env import SpaceShooterEnv
from space_shooter.rule_agent import RuleBasedAgent

agent = RuleBasedAgent()
counts = {
    "collected, shield was OFF (useful)": 0,
    "collected, shield was already ON (useless)": 0,
    "missed, shield was OFF (a REAL miss)": 0,
    "missed, shield was ON (ignored on purpose)": 0,
}

for seed in range(10):
    env = SpaceShooterEnv()
    env.reset(seed=seed)
    game = env.game
    done = False
    while not done:
        pickups_before = list(game.pickups)
        shield_before = game.shield
        collected_before = game.stats["shields_collected"]

        obs, reward, terminated, truncated, info = env.step(agent(game))
        done = terminated or truncated

        still_there = {id(p) for p in game.pickups}
        gone = [p for p in pickups_before if id(p) not in still_there]
        n_collected = game.stats["shields_collected"] - collected_before
        for i, _ in enumerate(gone):
            collected = i < n_collected
            if collected and not shield_before:
                counts["collected, shield was OFF (useful)"] += 1
            elif collected:
                counts["collected, shield was already ON (useless)"] += 1
            elif not shield_before:
                counts["missed, shield was OFF (a REAL miss)"] += 1
            else:
                counts["missed, shield was ON (ignored on purpose)"] += 1

print("Rings over 10 episodes of 5000 steps:")
for name, number in counts.items():
    print(f"  {name:45s}: {number}")