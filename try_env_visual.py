from space_shooter.env import SpaceShooterEnv

env = SpaceShooterEnv(render_mode="human", render_fps=60)
obs, _ = env.reset(seed=0)
for _ in range(900):                                    # about 15 seconds
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, _ = env.reset()
env.close()