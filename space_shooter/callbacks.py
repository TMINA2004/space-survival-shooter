
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


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

            # This environment is still playing
            if "episode" not in info:
                continue

            # Total reward (NOT discounted)
            total_reward = float(info["episode"]["r"])

            self.rewards.append(total_reward)

            # Record total episode reward
            self.logger.record(
                "episode/total_reward",
                total_reward
            )

            # Moving average reward
            self.logger.record(
                "episode/moving_avg_reward",
                float(np.mean(self.rewards[-self.window:]))
            )

            # Get game statistics
            stats = info.get("game_stats")

            if stats is not None:

                # Total kills
                self.logger.record(
                    "episode/kills",
                    stats["drifters_killed"]
                    + stats["hunters_killed"]
                )

                # Survival duration
                self.logger.record(
                    "episode/survival_steps",
                    info["game_steps"]
                )

                # Lives lost
                self.logger.record(
                    "episode/lives_lost",
                    stats["lives_lost"]
                )

                # Shooting accuracy
                if stats["shots_fired"] > 0:

                    self.logger.record(
                        "episode/accuracy",
                        stats["shots_hit"]
                        / stats["shots_fired"]
                    )

            # Write statistics to TensorBoard
            self.logger.dump(step=len(self.rewards))

        return True
