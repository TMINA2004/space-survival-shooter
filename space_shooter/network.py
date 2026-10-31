
import torch
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpaceFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom neural network for our 25-number observation.

    obs[0:6]   -> Ship and status
    obs[6:22]  -> 4 enemy slots x 4 values
    obs[22:25] -> Nearest pickup

    Each section has its own small neural network.
    The enemy network shares weights across all 4 enemies.
    The results are concatenated and passed through
    a final network to produce features_dim features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128
    ):

        super().__init__(observation_space, features_dim)

        # Ship status network: 6 -> 32
        self.status_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU()
        )

        # Enemy network: 4 -> 32
        self.enemy_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Pickup network: 3 -> 16
        self.pickup_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU()
        )

        # Combine all features:
        # 32 + (4 * 32) + 16 = 176
        self.mix = nn.Sequential(
            nn.Linear(176, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        batch = obs.shape[0]

        # Process ship status
        status = self.status_net(obs[:, 0:6])

        # Process four enemies using shared weights
        enemies = self.enemy_net(
            obs[:, 6:22].reshape(batch, 4, 4)
        )

        # Flatten enemy features:
        # batch x 4 x 32 -> batch x 128
        enemies = enemies.reshape(batch, -1)

        # Process nearest pickup
        pickup = self.pickup_net(obs[:, 22:25])

        # Concatenate all features
        combined = torch.cat(
            [status, enemies, pickup],
            dim=1
        )

        # Final feature representation
        return self.mix(combined)
