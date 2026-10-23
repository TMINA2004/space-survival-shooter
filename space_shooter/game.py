from dataclasses import dataclass

import numpy as np

from .config import Config

# The 6 actions
STAY, UP, DOWN, LEFT, RIGHT, SHOOT = range(6)

# Enemy types
DRIFTER, HUNTER = 0, 1


@dataclass
class Enemy:
    x: float
    y: float
    kind: int = DRIFTER
    hp: int = 1


class Game:
    """The rules of the game. No graphics, no reward, only the game world."""

    def __init__(self, config=None, seed=None):
        self.cfg = config or Config()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        size = self.cfg.arena_size
        self.ship_x = size / 2
        self.ship_y = size - self.cfg.ship_start_offset
        self.lives = self.cfg.start_lives
        self.invulnerable = 0
        self.steps = 0
        self.game_over = False
        self.enemies = []
        self.spawn_timer = self.spawn_interval()

    def step(self, action):
        events = {"lives_lost": 0, "game_over": False}

        if self.game_over:
            return events

        if self.invulnerable > 0:
            self.invulnerable -= 1

        self._move_ship(action)
        self._move_enemies()
        self._handle_ship_collisions(events)
        self._remove_offscreen_enemies()
        self._spawn_enemies()

        self.steps += 1
        if self.lives <= 0:
            self.game_over = True
            events["game_over"] = True

        return events

    def spawn_interval(self):
        interval = self.cfg.spawn_start_interval - self.steps / 200
        return int(max(self.cfg.spawn_min_interval, interval))

    def _move_ship(self, action):
        speed = self.cfg.ship_speed

        if action == UP:
            self.ship_y -= speed
        elif action == DOWN:
            self.ship_y += speed
        elif action == LEFT:
            self.ship_x -= speed
        elif action == RIGHT:
            self.ship_x += speed

        r = self.cfg.ship_radius
        size = self.cfg.arena_size
        self.ship_x = min(max(self.ship_x, r), size - r)
        self.ship_y = min(max(self.ship_y, r), size - r)

    def _move_enemies(self):
        for enemy in self.enemies:
            if enemy.kind == DRIFTER:
                enemy.y += self.cfg.drifter_speed

    def _handle_ship_collisions(self, events):
        hit_distance = self.cfg.ship_radius + self.cfg.enemy_radius
        survivors = []

        for enemy in self.enemies:
            dx = enemy.x - self.ship_x
            dy = enemy.y - self.ship_y

            if dx * dx + dy * dy < hit_distance ** 2:
                if self.invulnerable == 0:
                    self.lives -= 1
                    events["lives_lost"] += 1
                    self.invulnerable = self.cfg.invulnerable_steps
                # The enemy is destroyed even when the ship is protected.
            else:
                survivors.append(enemy)

        self.enemies = survivors

    def _remove_offscreen_enemies(self):
        limit = self.cfg.arena_size + self.cfg.enemy_radius
        self.enemies = [e for e in self.enemies if e.y <= limit]

    def _spawn_enemies(self):
        self.spawn_timer -= 1

        if self.spawn_timer <= 0:
            r = self.cfg.enemy_radius
            x = float(self.rng.uniform(r, self.cfg.arena_size - r))
            self.enemies.append(
                Enemy(x=x, y=-r, kind=DRIFTER, hp=1)
            )
            self.spawn_timer = self.spawn_interval()