import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Config, RewardConfig
from .game import Game, HUNTER, N_ACTIONS

N_ENEMIES_IN_STATE = 4
OBS_SIZE = 6 + 4 * N_ENEMIES_IN_STATE + 3        # = 25


class SpaceShooterEnv(gym.Env):
    """Gymnasium wrapper around the Game.

    render_mode=None     -> HEADLESS: no window, used for training and testing.
    render_mode="human"  -> INTERACTIVE view: a Pygame window shows the game.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config=None, reward_config=None, render_mode=None, render_fps=60):
        super().__init__()
        self.cfg = config or Config()
        self.rewards = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.game = Game(self.cfg)
        self._renderer = None

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # The two functions every Gymnasium environment must have
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)          # seed=None keeps the same random stream
        if self.render_mode == "human":
            self.render()
        return self._observation(), {}

    def step(self, action):
        events = self.game.step(action)
        reward = self._reward(events)

        terminated = self.game.game_over                                   # lost all lives
        truncated = (not terminated) and self.game.steps >= self.cfg.max_episode_steps

        info = {}
        if terminated or truncated:                                        # episode summary
            info["game_stats"] = dict(self.game.stats)
            info["game_steps"] = self.game.steps

        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering (the window is created only when it is really needed)
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self._renderer is None:
            from .renderer import Renderer        # pygame is imported only here
            self._renderer = Renderer(self.cfg, fps=self.render_fps)
        import pygame
        pygame.event.pump()                       # keeps the window responsive
        self._renderer.draw(self.game)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Reward: turns the game's EVENTS into a number
    # ------------------------------------------------------------------
    def _reward(self, events):
        rw = self.rewards
        reward = rw.survive
        reward += rw.drifter_killed * events["drifters_killed"]
        reward += rw.hunter_killed * events["hunters_killed"]
        reward += rw.life_lost * events["lives_lost"]
        reward += rw.shield_collected * events["shields_collected"]
        return float(reward)

    # ------------------------------------------------------------------
    # Observation: turns the game world into 25 numbers
    # ------------------------------------------------------------------
    def _observation(self):
        g, cfg = self.game, self.cfg
        size = cfg.arena_size
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # Ship and status (indices 0-5)
        obs[0] = g.ship_x / size
        obs[1] = g.ship_y / size
        obs[2] = g.lives / cfg.start_lives
        obs[3] = g.ammo / cfg.max_ammo if cfg.ammo_limited else 1.0
        obs[4] = 1.0 if g.shield else 0.0
        obs[5] = g.shoot_cooldown / cfg.shoot_cooldown

        # The 4 nearest enemies (indices 6-21), nearest first
        def squared_distance(obj):
            return (obj.x - g.ship_x) ** 2 + (obj.y - g.ship_y) ** 2

        nearest = sorted(g.enemies, key=squared_distance)[:N_ENEMIES_IN_STATE]
        for i, enemy in enumerate(nearest):
            base = 6 + 4 * i
            obs[base] = (enemy.x - g.ship_x) / size
            obs[base + 1] = (enemy.y - g.ship_y) / size
            obs[base + 2] = 1.0 if enemy.kind == HUNTER else 0.0
            obs[base + 3] = 1.0

        # The nearest shield pickup (indices 22-24)
        if g.pickups:
            pickup = min(g.pickups, key=squared_distance)
            obs[22] = (pickup.x - g.ship_x) / size
            obs[23] = (pickup.y - g.ship_y) / size
            obs[24] = 1.0

        return np.clip(obs, -1.0, 1.0)