
import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

from .evaluation import play_episode
from .rl_player import RLPlayer


class EpisodeStatsCallback(BaseCallback):
    """
    For EVERY finished episode, saves its total reward
    and game statistics to TensorBoard.
    """

    def __init__(self, window=50):
        super().__init__()
        self.window = window
        self.rewards = []

    def _on_step(self):
        for info in self.locals["infos"]:

            # If the episode is not finished, skip it.
            if "episode" not in info:
                continue

            # Total reward (NOT discounted).
            total_reward = float(info["episode"]["r"])
            self.rewards.append(total_reward)

            self.logger.record(
                "episode/total_reward",
                total_reward
            )

            self.logger.record(
                "episode/moving_avg_reward",
                float(np.mean(self.rewards[-self.window:]))
            )

            # Game statistics.
            stats = info.get("game_stats")

            if stats is not None:
                self.logger.record(
                    "episode/kills",
                    stats["drifters_killed"] + stats["hunters_killed"]
                )

                self.logger.record(
                    "episode/survival_steps",
                    info["game_steps"]
                )

                self.logger.record(
                    "episode/lives_lost",
                    stats["lives_lost"]
                )

                if stats["shots_fired"] > 0:
                    self.logger.record(
                        "episode/accuracy",
                        stats["shots_hit"] / stats["shots_fired"]
                    )

            # X-axis = episode number.
            self.logger.dump(step=len(self.rewards))

        return True


class PeriodicEvalCallback(BaseCallback):
    """
    Every eval_every steps, plays the SAME validation games
    with the current model.

    Logs results to TensorBoard, prints one line,
    and keeps the best model so far.
    """

    def __init__(
        self,
        save_dir,
        eval_every=100_000,
        seeds=range(2000, 2010),
        config=None
    ):
        super().__init__()

        self.save_dir = save_dir
        self.eval_every = eval_every
        self.seeds = list(seeds)
        self.config = config

        self.next_eval = eval_every
        self.best_reward = -float("inf")

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            self._evaluate()

        return True

    def _evaluate(self):
        player = RLPlayer(
            self.model,
            deterministic=True
        )

        rows = [
            play_episode(player, seed, self.config)
            for seed in self.seeds
        ]

        def mean(key):
            return float(np.mean([row[key] for row in rows]))

        reward = mean("total_reward")
        survival = mean("survival_steps")
        survival_rate = mean("survived")
        kills = mean("kills")

        self.logger.record("eval/mean_reward", reward)

        self.logger.record(
            "eval/mean_survival_steps",
            survival
        )

        self.logger.record(
            "eval/survival_rate",
            survival_rate
        )

        self.logger.record(
            "eval/mean_kills",
            kills
        )

        self.logger.dump(step=self.num_timesteps)

        # Save the best model so far.
        is_best = reward > self.best_reward

        if is_best:
            self.best_reward = reward

            self.model.save(
                os.path.join(self.save_dir, "best_model")
            )

        marker = "  <- new best" if is_best else ""

        print(
            f"[{self.num_timesteps:>10,} steps] "
            f"validation: reward {reward:7.1f} | "
            f"survival {survival:6.0f} | "
            f"kills {kills:5.1f}{marker}",
            flush=True
        )
