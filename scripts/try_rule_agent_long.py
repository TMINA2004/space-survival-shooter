from space_shooter.config import Config
from space_shooter.evaluation import play_episode
from space_shooter.rule_agent import RuleBasedAgent

agent = RuleBasedAgent()
config = Config(max_episode_steps=20_000)        # 4 times longer than usual

for seed in range(6):
    row = play_episode(agent, seed, config)
    print(f"seed {seed}: survived {row['survival_steps']:6d} steps | "
          f"lives lost = {row['lives_lost']} | kills = {row['kills']} | "
          f"reward = {row['total_reward']:.1f}")