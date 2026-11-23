
import torch

from space_shooter.env import SpaceShooterEnv
from space_shooter.network import SpaceFeaturesExtractor


# Create the game environment
env = SpaceShooterEnv()

# Create the custom neural network
net = SpaceFeaturesExtractor(
    env.observation_space,
    features_dim=128
)


# ==========================================
# TEST 1: Check output shape
# ==========================================

obs = torch.rand(5, 25) * 2 - 1

out = net(obs)

assert out.shape == (5, 128)

print(
    "Test 1 OK: output shape",
    tuple(out.shape)
)


# ==========================================
# TEST 2: Check gradients
# ==========================================

out.sum().backward()

assert all(
    p.grad is not None
    for p in net.parameters()
)

print("Test 2 OK: gradients reach every parameter")


# ==========================================
# TEST 3: Check shared enemy weights
# ==========================================

x = torch.zeros(1, 25)

# Enemy in slot 1
x[0, 6:10] = torch.tensor([
    0.1, -0.2, 1.0, 1.0
])

# Same enemy in slot 2
x[0, 10:14] = torch.tensor([
    0.1, -0.2, 1.0, 1.0
])

# Encode all four enemies
encoded = net.enemy_net(
    x[:, 6:22].reshape(1, 4, 4)
)

# Identical enemies should have identical encodings
assert torch.allclose(
    encoded[0, 0],
    encoded[0, 1]
)

print(
    "Test 3 OK: identical enemies in two slots "
    "are encoded identically (shared weights)"
)


# ==========================================
# COUNT NETWORK PARAMETERS
# ==========================================

total_params = sum(
    p.numel()
    for p in net.parameters()
)

print(
    "Parameters in our custom network:",
    total_params
)

env.close()
