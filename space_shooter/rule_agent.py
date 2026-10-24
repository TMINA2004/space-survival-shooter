import math
from dataclasses import dataclass

from .game import STAY, UP, DOWN, LEFT, RIGHT, HUNTER, make_action

# Order matters: when two moves are equally good, the first one in this list wins.
MOVES = [STAY, LEFT, RIGHT, DOWN, UP]


@dataclass
class RuleParams:
    """All the numbers of the rule-based agent (so we can tune them on Game Day 11)."""

    horizon: int = 15                 # steps we look ahead when checking danger
    danger_distance: float = 44.0     # collision is 30 px, plus a safety margin of 14
    look_range: float = 180.0         # farther enemies cannot reach us within the horizon
    align_tolerance: float = 8.0      # "aligned" when walking: |dx| below this
    shoot_tolerance: float = 13.0     # a bullet hits if |dx| < 19: we keep a margin
    ammo_reserve: int = 5             # bullets kept for Hunters
    shield_range: float = 200.0       # chase a pickup only when it is this close (vertical)
    home_x: float = 300.0
    home_y: float = 560.0


def move_ship(x, y, move, cfg):
    """Where the ship ends up after one move (same rules as the game, walls included)."""
    speed = cfg.ship_speed
    if move == UP:
        y -= speed
    elif move == DOWN:
        y += speed
    elif move == LEFT:
        x -= speed
    elif move == RIGHT:
        x += speed
    r, size = cfg.ship_radius, cfg.arena_size
    return min(max(x, r), size - r), min(max(y, r), size - r)


class RuleBasedAgent:
    """A hand-made player. Each step: choose a MOVE by priority rules, then decide to SHOOT.

    It is used like the other reference players: agent(game, rng) -> action (0-9).
    """

    def __init__(self, params=None):
        self.p = params or RuleParams()

    def __call__(self, game, rng=None):
        move = self._choose_move(game)
        shoot = self._should_shoot(game, move)
        return make_action(move, shoot)

    # ------------------------------------------------------------------
    # MOVE: rules in priority order
    # ------------------------------------------------------------------
    def _choose_move(self, game):
        # RULE 1: dodge
        nearby = self._nearby_enemies(game)
        if nearby:
            margins = {m: self._future_min_distance(game, nearby, m) for m in MOVES}
            if margins[STAY] < self.p.danger_distance:
                return max(MOVES, key=lambda m: margins[m])     # safest move

        # RULE 2: collect a shield
        move = self._go_for_shield(game)
        if move is not None:
            return move

        # RULE 3: hunt a target
        move = self._go_for_target(game)
        if move is not None:
            return move

        # RULE 4: go home
        return self._go_home(game)

    def _nearby_enemies(self, game):
        limit = self.p.look_range
        return [[e.x, e.y, e.kind] for e in game.enemies
                if abs(e.x - game.ship_x) < limit and abs(e.y - game.ship_y) < limit]

    def _future_min_distance(self, game, nearby, move):
        """If we repeat `move` for `horizon` steps, how close will an enemy get to us?"""
        cfg = game.cfg
        x, y = game.ship_x, game.ship_y
        enemies = [list(e) for e in nearby]                     # copies we can move freely
        closest = float("inf")
        for _ in range(self.p.horizon):
            x, y = move_ship(x, y, move, cfg)
            for e in enemies:
                if e[2] == HUNTER:                              # a Hunter walks toward us
                    dx, dy = x - e[0], y - e[1]
                    distance = math.hypot(dx, dy)
                    if distance > 0:
                        e[0] += cfg.hunter_speed * dx / distance
                        e[1] += cfg.hunter_speed * dy / distance
                else:                                           # a Drifter falls straight down
                    e[1] += cfg.drifter_speed
                closest = min(closest, math.hypot(e[0] - x, e[1] - y))
        return closest

    def _go_for_shield(self, game):
        if game.shield or not game.pickups:
            return None
        cfg = game.cfg
        best = None
        for pickup in game.pickups:
            height = game.ship_y - pickup.y                     # how far above us it is
            if height < 10 or height > self.p.shield_range:
                continue
            dx = pickup.x - game.ship_x
            time_to_arrive = height / cfg.pickup_speed
            time_to_align = max(0.0, abs(dx) - 5) / cfg.ship_speed
            if time_to_align + 3 <= time_to_arrive:             # we can make it in time
                if best is None or time_to_arrive < best[0]:
                    best = (time_to_arrive, dx)
        if best is None:
            return None
        dx = best[1]
        if abs(dx) > 5:
            return RIGHT if dx > 0 else LEFT
        return STAY                                             # aligned: let it fall on us

    def _go_for_target(self, game):
        cfg = game.cfg
        if not cfg.shooting_enabled:
            return None
        if cfg.ammo_limited and game.ammo < 1:
            return None

        sx, sy = game.ship_x, game.ship_y
        above = [e for e in game.enemies if e.y < sy - 20]
        hunters = [e for e in above if e.kind == HUNTER]

        if hunters:
            candidates = hunters                                # Hunters first, always
        else:
            if cfg.ammo_limited and game.ammo <= self.p.ammo_reserve:
                return None                                     # keep the reserve
            # Drifter: can we align AND hit it before it reaches us?
            candidates = [e for e in above if (sy - e.y) - 0.5 * abs(e.x - sx) > 60]
            if not candidates:
                return None

        target = min(candidates, key=lambda e: math.hypot(e.x - sx, e.y - sy))
        dx = target.x - sx
        if abs(dx) > self.p.align_tolerance:
            return RIGHT if dx > 0 else LEFT
        return STAY                                             # aligned: wait and shoot

    def _go_home(self, game):
        p = self.p
        if game.ship_y < p.home_y - 6:
            return DOWN
        if game.ship_x < p.home_x - 12:
            return RIGHT
        if game.ship_x > p.home_x + 12:
            return LEFT
        return STAY

    # ------------------------------------------------------------------
    # SHOOT: decided on its own, so the agent can move AND shoot
    # ------------------------------------------------------------------
    def _should_shoot(self, game, move):
        cfg = game.cfg
        if not cfg.shooting_enabled or game.shoot_cooldown > 1:     # still cooling down
            return False
        if cfg.ammo_limited and game.ammo < 1:
            return False

        sx, sy = move_ship(game.ship_x, game.ship_y, move, cfg)     # where we will be
        in_line = [e for e in game.enemies
                   if e.y < sy - cfg.ship_radius and abs(e.x - sx) < self.p.shoot_tolerance]
        if not in_line:
            return False
        if any(e.kind == HUNTER for e in in_line):
            return True
        return (not cfg.ammo_limited) or game.ammo > self.p.ammo_reserve