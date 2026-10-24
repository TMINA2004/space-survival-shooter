import sys
import gymnasium, pygame, numpy, torch, stable_baselines3, pandas, matplotlib

print("Python      :", sys.version.split()[0])
print("Gymnasium   :", gymnasium.__version__)
print("Pygame      :", pygame.version.ver)
print("NumPy       :", numpy.__version__)
print("PyTorch     :", torch.__version__)
print("SB3         :", stable_baselines3.__version__)
print("Pandas      :", pandas.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("GPU found   :", torch.cuda.is_available())

# Mini RL test: a robot learns to balance a pole for 5000 steps
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "CartPole-v1", verbose=0)
model.learn(total_timesteps=5000)
print("RL works! Your first agent was trained.")