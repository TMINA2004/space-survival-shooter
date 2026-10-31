import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Config, RewardConfig
from .game import Game, HUNTER, N_ACTIONS


N_ENEMIES_IN_STATE = 4
OBS_SIZE = 6 + 4 * N_ENEMIES_IN_STATE + 3  # 25


def build_observation(game):
    """Turns a Game into the 25 numbers the agent sees."""

    g, cfg = game, game.cfg
    size = cfg.arena_size

    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # Ship and status (indices 0-5)
    obs[0] = g.ship_x / size
    obs[1] = g.ship_y / size
    obs[2] = g.lives / cfg.start_lives

    obs[3] = (
        g.ammo / cfg.max_ammo
        if cfg.ammo_limited
        else 1.0
    )

    obs[4] = 1.0 if g.shield else 0.0
    obs[5] = g.shoot_cooldown / cfg.shoot_cooldown

    # Calculate squared distance
    def squared_distance(obj):
        return (
            (obj.x - g.ship_x) ** 2
            + (obj.y - g.ship_y) ** 2
        )

    # The 4 nearest enemies (indices 6-21)
    nearest = sorted(
        g.enemies,
        key=squared_distance
    )[:N_ENEMIES_IN_STATE]

    for i, enemy in enumerate(nearest):
        base = 6 + 4 * i

        obs[base] = (enemy.x - g.ship_x) / size
        obs[base + 1] = (enemy.y - g.ship_y) / size
        obs[base + 2] = (
            1.0 if enemy.kind == HUNTER else 0.0
        )
        obs[base + 3] = 1.0

    # Nearest shield pickup (indices 22-24)
    if g.pickups:
        pickup = min(
            g.pickups,
            key=squared_distance
        )

        obs[22] = (pickup.x - g.ship_x) / size
        obs[23] = (pickup.y - g.ship_y) / size
        obs[24] = 1.0

    return np.clip(obs, -1.0, 1.0)


class SpaceShooterEnv(gym.Env):

    """
    Gymnasium wrapper around the Game.

    render_mode=None:
        Headless mode for training and testing.

    render_mode="human":
        Interactive Pygame window.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 60
    }

    def __init__(
        self,
        config=None,
        reward_config=None,
        render_mode=None,
        render_fps=60
    ):

        super().__init__()

        self.cfg = config or Config()
        self.rewards = reward_config or RewardConfig()

        self.render_mode = render_mode
        self.render_fps = render_fps

        self.game = Game(self.cfg)
        self._renderer = None

        self.action_space = spaces.Discrete(N_ACTIONS)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.game.reset(seed=seed)

        if self.render_mode == "human":
            self.render()

        return self._observation(), {}

    def step(self, action):

        had_shield = self.game.shield

        events = self.game.step(action)

        reward = self._reward(events, had_shield)

        terminated = self.game.game_over

        truncated = (
            (not terminated)
            and self.game.steps >= self.cfg.max_episode_steps
        )

        info = {}

        if terminated or truncated:
            info["game_stats"] = dict(self.game.stats)
            info["game_steps"] = self.game.steps

        if self.render_mode == "human":
            self.render()

        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            info
        )

    def render(self):

        if self.render_mode != "human":
            return

        if self._renderer is None:
            from .renderer import Renderer
            self._renderer = Renderer(
                self.cfg,
                fps=self.render_fps
            )

        import pygame

        pygame.event.pump()
        self._renderer.draw(self.game)

    def close(self):

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _reward(self, events, had_shield=False):

        rw = self.rewards

        reward = rw.survive

        reward += (
            rw.drifter_killed * events["drifters_killed"]
        )

        reward += (
            rw.hunter_killed * events["hunters_killed"]
        )

        reward += (
            rw.life_lost * events["lives_lost"]
        )

        # Shield reward only when the player had no shield
        if not had_shield:
            reward += (
                rw.shield_collected
                * min(events["shields_collected"], 1)
            )

        return float(reward)

    def _observation(self):
        return build_observation(self.game)