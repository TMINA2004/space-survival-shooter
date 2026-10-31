
from .env import build_observation


class RLPlayer:
    """
    Wraps a trained Stable-Baselines3 model as a player.

    player(game, rng) -> action
    """

    def __init__(self, model, deterministic=True):
        self.model = model
        self.deterministic = deterministic

    def __call__(self, game, rng=None):

        # Convert game state into the observation
        obs = build_observation(game)

        # Ask the trained model to choose an action
        action, _ = self.model.predict(
            obs,
            deterministic=self.deterministic
        )

        return int(action)
