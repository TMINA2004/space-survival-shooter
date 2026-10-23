import numpy as np

from .config import Config

# The 6 actions
STAY, UP, DOWN, LEFT, RIGHT, SHOOT = range(6)


class Game:
    """The rules of the game. No graphics, no reward, only the game world."""

    def __init__(self, config=None, seed=None):
        self.cfg = config or Config()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        """Start a new game."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        size = self.cfg.arena_size
        self.ship_x = size / 2
        self.ship_y = size - self.cfg.ship_start_offset
        self.lives = self.cfg.start_lives
        self.steps = 0
        self.game_over = False

    def step(self, action):
        """Advance the game by ONE tick."""
        events = {}

        if self.game_over:
            return events

        self._move_ship(action)
        self.steps += 1
        return events

    def _move_ship(self, action):
        speed = self.cfg.ship_speed

        # x grows to the RIGHT; y grows DOWNWARD.
        if action == UP:
            self.ship_y -= speed
        elif action == DOWN:
            self.ship_y += speed
        elif action == LEFT:
            self.ship_x -= speed
        elif action == RIGHT:
            self.ship_x += speed

        # Keep the ship inside the arena.
        r = self.cfg.ship_radius
        size = self.cfg.arena_size

        self.ship_x = min(max(self.ship_x, r), size - r)
        self.ship_y = min(max(self.ship_y, r), size - r)