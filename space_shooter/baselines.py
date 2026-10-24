"""Simple reference players. Each one is a function: policy(game, rng) -> action (0-9)."""

from .game import STAY, make_action


def stay(game, rng):
    """Does nothing."""
    return make_action(STAY, False)


def stay_and_shoot(game, rng):
    """Stands still and holds the fire button."""
    return make_action(STAY, True)


def random_player(game, rng):
    """Presses random keys."""
    return int(rng.integers(0, 10))


def column_shooter(game, rng):
    """Stands still and shoots only when an enemy is above it, in its column."""
    threat = any(abs(e.x - game.ship_x) < 25 and e.y < game.ship_y for e in game.enemies)
    return make_action(STAY, threat)


REFERENCE_PLAYERS = {
    "STAY": stay,
    "STAY + SHOOT": stay_and_shoot,
    "RANDOM": random_player,
    "COLUMN SHOOTER": column_shooter,
}