import math
from dataclasses import dataclass

import numpy as np

from .config import Config

# The MOVE part of an action
STAY, UP, DOWN, LEFT, RIGHT = range(5)
N_MOVES = 5
N_ACTIONS = 10            # 5 moves x (no shoot / shoot)

# Enemy types
DRIFTER, HUNTER = 0, 1


def make_action(move, shoot=False):
    """Combine a move (0-4) and a shoot flag into one action (0-9)."""
    return move + N_MOVES * int(bool(shoot))


def split_action(action):
    """Split an action (0-9) into (move, shoot)."""
    return action % N_MOVES, (action // N_MOVES) == 1


@dataclass
class Enemy:
    x: float
    y: float
    kind: int = DRIFTER
    hp: int = 1


@dataclass
class Bullet:
    x: float
    y: float


@dataclass
class Pickup:
    x: float
    y: float


class Game:
    """The rules of the game. No graphics, no reward, only the game world."""

    def __init__(self, config=None, seed=None):
        self.cfg = config or Config()
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Start a new game
    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        size = self.cfg.arena_size
        self.ship_x = size / 2                                   # bottom center
        self.ship_y = size - self.cfg.ship_start_offset
        self.lives = self.cfg.start_lives
        self.shield = False
        self.invulnerable = 0                                    # steps of protection left
        self.shoot_cooldown = 0                                  # steps before next shot
        self.ammo = self.cfg.max_ammo                            # bullets in the magazine
        self.ammo_timer = self.cfg.ammo_recharge_steps           # steps until +1 ammo
        self.steps = 0
        self.game_over = False
        self.enemies = []
        self.bullets = []
        self.pickups = []
        self.spawn_timer = self.spawn_interval()
        self.pickup_timer = self._pickup_delay() if self.cfg.shields_enabled else 0

        # Running totals for the whole game (used for evaluation later)
        self.stats = {
            "shots_fired": 0,
            "shots_hit": 0,
            "dry_fires": 0,
            "drifters_killed": 0,
            "hunters_killed": 0,
            "lives_lost": 0,
            "shields_collected": 0,
            "hits_blocked": 0,
        }

    # ------------------------------------------------------------------
    # One tick of the game
    # ------------------------------------------------------------------
    def step(self, action):
        """Advance the game by ONE tick. Returns a dict of what happened."""
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action must be between 0 and {N_ACTIONS - 1}, got {action}")

        events = {
            "lives_lost": 0,
            "shots_fired": 0,
            "shots_hit": 0,
            "dry_fires": 0,
            "drifters_killed": 0,
            "hunters_killed": 0,
            "shields_collected": 0,
            "hits_blocked": 0,
            "game_over": False,
        }

        if self.game_over:
            return events

        move, shoot = split_action(action)

        if self.invulnerable > 0:
            self.invulnerable -= 1
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        self._recharge_ammo()

        self._move_ship(move)
        self._handle_shooting(shoot, events)
        self._move_bullets()
        self._move_enemies()
        self._move_pickups()
        self._handle_bullet_hits(events)
        self._handle_pickups(events)
        self._handle_ship_collisions(events)
        self._remove_offscreen_enemies()
        self._spawn_enemies()
        self._spawn_pickups()

        self.steps += 1
        for key in self.stats:                 # add this step's events to the totals
            self.stats[key] += events[key]

        if self.lives <= 0:
            self.game_over = True
            events["game_over"] = True
        return events

    # ------------------------------------------------------------------
    # Difficulty
    # ------------------------------------------------------------------
    def spawn_interval(self):
        """Steps between two enemies. Gets smaller with time (harder game)."""
        interval = self.cfg.spawn_start_interval - self.steps / 200
        return int(max(self.cfg.spawn_min_interval, interval))

    def _pickup_delay(self):
        """Steps until the next shield pickup: random, around pickup_every_steps."""
        every = self.cfg.pickup_every_steps
        return int(self.rng.integers(int(0.75 * every), int(1.25 * every) + 1))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _move_ship(self, move):
        speed = self.cfg.ship_speed

        # Screen coordinates: x grows to the RIGHT, y grows DOWNWARD.
        if move == UP:
            self.ship_y -= speed
        elif move == DOWN:
            self.ship_y += speed
        elif move == LEFT:
            self.ship_x -= speed
        elif move == RIGHT:
            self.ship_x += speed

        r = self.cfg.ship_radius
        size = self.cfg.arena_size
        self.ship_x = min(max(self.ship_x, r), size - r)
        self.ship_y = min(max(self.ship_y, r), size - r)

    def _handle_shooting(self, shoot, events):
        if not shoot or not self.cfg.shooting_enabled or self.shoot_cooldown > 0:
            return

        if self.cfg.ammo_limited:
            if self.ammo <= 0:
                events["dry_fires"] += 1        # tried to shoot with an empty magazine
                return
            self.ammo -= 1

        tip_y = self.ship_y - self.cfg.ship_radius              # bullet leaves from the ship's tip
        self.bullets.append(Bullet(x=self.ship_x, y=tip_y))
        self.shoot_cooldown = self.cfg.shoot_cooldown
        events["shots_fired"] += 1

    def _recharge_ammo(self):
        if not self.cfg.ammo_limited:
            return
        if self.ammo >= self.cfg.max_ammo:                      # full: timer waits
            self.ammo_timer = self.cfg.ammo_recharge_steps
            return
        self.ammo_timer -= 1
        if self.ammo_timer <= 0:
            self.ammo += 1
            self.ammo_timer = self.cfg.ammo_recharge_steps

    def _move_bullets(self):
        for bullet in self.bullets:
            bullet.y -= self.cfg.bullet_speed
        limit = -self.cfg.bullet_radius
        self.bullets = [b for b in self.bullets if b.y >= limit]   # drop bullets that left the arena

    def _move_enemies(self):
        for enemy in self.enemies:
            if enemy.kind == DRIFTER:
                enemy.y += self.cfg.drifter_speed               # falls straight down
            else:                                               # HUNTER: walks toward the ship
                dx = self.ship_x - enemy.x
                dy = self.ship_y - enemy.y
                distance = math.hypot(dx, dy)
                if distance > 0:
                    enemy.x += self.cfg.hunter_speed * dx / distance
                    enemy.y += self.cfg.hunter_speed * dy / distance

    def _move_pickups(self):
        for pickup in self.pickups:
            pickup.y += self.cfg.pickup_speed
        limit = self.cfg.arena_size + self.cfg.pickup_radius
        self.pickups = [p for p in self.pickups if p.y <= limit]

    def _handle_bullet_hits(self, events):
        hit_distance = self.cfg.bullet_radius + self.cfg.enemy_radius
        remaining_bullets = []

        for bullet in self.bullets:
            hit = False
            for enemy in self.enemies:
                dx = enemy.x - bullet.x
                dy = enemy.y - bullet.y
                if dx * dx + dy * dy < hit_distance ** 2:
                    enemy.hp -= 1
                    events["shots_hit"] += 1
                    hit = True
                    break                       # one bullet hits only one enemy
            if not hit:
                remaining_bullets.append(bullet)
        self.bullets = remaining_bullets

        alive = []
        for enemy in self.enemies:
            if enemy.hp > 0:
                alive.append(enemy)
            elif enemy.kind == DRIFTER:
                events["drifters_killed"] += 1
            else:
                events["hunters_killed"] += 1
        self.enemies = alive

    def _handle_pickups(self, events):
        reach = self.cfg.ship_radius + self.cfg.pickup_radius
        remaining = []
        for pickup in self.pickups:
            dx = pickup.x - self.ship_x
            dy = pickup.y - self.ship_y
            if dx * dx + dy * dy < reach ** 2:
                self.shield = True
                events["shields_collected"] += 1
            else:
                remaining.append(pickup)
        self.pickups = remaining

    def _handle_ship_collisions(self, events):
        hit_distance = self.cfg.ship_radius + self.cfg.enemy_radius
        survivors = []
        for enemy in self.enemies:
            dx = enemy.x - self.ship_x
            dy = enemy.y - self.ship_y
            if dx * dx + dy * dy < hit_distance ** 2:       # circles overlap
                if self.invulnerable == 0:
                    if self.shield:
                        self.shield = False                 # the shield takes the hit
                        events["hits_blocked"] += 1
                    else:
                        self.lives -= 1
                        events["lives_lost"] += 1
                    self.invulnerable = self.cfg.invulnerable_steps
                # the enemy is destroyed either way (we do not keep it)
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
            is_hunter = self.cfg.hunters_enabled and self.rng.random() < self.cfg.hunter_probability
            if is_hunter:
                self.enemies.append(Enemy(x=x, y=-r, kind=HUNTER, hp=self.cfg.hunter_hp))
            else:
                self.enemies.append(Enemy(x=x, y=-r, kind=DRIFTER, hp=1))
            self.spawn_timer = self.spawn_interval()

    def _spawn_pickups(self):
        if not self.cfg.shields_enabled:
            return
        self.pickup_timer -= 1
        if self.pickup_timer <= 0:
            r = self.cfg.pickup_radius
            x = float(self.rng.uniform(r, self.cfg.arena_size - r))
            self.pickups.append(Pickup(x=x, y=-r))
            self.pickup_timer = self._pickup_delay()