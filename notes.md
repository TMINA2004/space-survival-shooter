# Day 1 — Game Design Questions

## 1. How could the agent cheat the reward system?

The agent could stay in a safe corner to collect survival rewards without fighting enemies. To prevent this, the game can gradually increase the difficulty and give the agent stronger reasons to avoid enemies and survive actively.

## 2. Why use relative positions instead of absolute coordinates?

Relative positions tell the agent where enemies are in relation to the spaceship. This helps the agent learn how to react to nearby threats, even when the spaceship and enemies are in different parts of the screen.

## 3. What is the difference between termination and truncation?

Termination happens when the spaceship loses all its lives. Truncation happens when the episode reaches the maximum limit of 5,000 steps, even if the spaceship is still alive.

## 4. How could we simplify the observation space?

We could remove information about the fourth-nearest enemy. This would reduce the number of observations, but the agent would have less information about distant threats and might make worse decisions.

# Day 2 — Folders, Config, and the Game Core

### 1. Why do we keep all numbers in `config.py`?

We keep the game parameters in `config.py` so we can change them in one place. This makes the code easier to maintain and helps us run experiments with different settings.

### 2. Why must `game.py` never import `pygame`?

The game logic should work independently of graphics. This allows the RL agent to train without rendering a window, making training faster and keeping the game rules separate from the visual presentation.

### 3. In screen coordinates, what happens to `y` when the ship goes up?

The `y` coordinate decreases because the origin `(0, 0)` is at the top-left corner of the screen.

### 4. Why does `step()` return events and not a reward?

The game logic reports what happens during each step. The RL environment calculates the reward separately, allowing us to change the reward system without modifying the game rules.

### 5. Why do we test the walls and corners?

We test the walls and corners to make sure the ship stays inside the arena. These tests help detect movement and boundary bugs before we add more game features.

## Day 3 — Enemies, collisions, and difficulty

### 1. Why do we use invulnerability frames?
Invulnerability frames prevent the player from losing multiple lives in consecutive steps while overlapping with an enemy. This makes collisions fairer and gameplay more manageable.

### 2. Why destroy an enemy even when the ship is invulnerable?
Destroying the enemy prevents it from remaining on top of the ship and causing repeated collisions. It also removes the enemy from the game.

### 3. Why use random seeds?
Random seeds make experiments reproducible. Using the same seed generates the same random enemy behavior, allowing fair comparisons between different agents.

### 4. Why remove enemies that leave the screen?
Enemies that leave the screen can no longer interact with the ship. Removing them prevents the enemy list from growing indefinitely and keeps the simulation efficient.

### 5. What are the baseline results?
- **STAY:** 1,353 average survival steps.
- **RANDOM:** 1,027 average survival steps.
- **Simulation speed:** approximately 101,132 steps per second during the Day 3 benchmark.

In this experiment, STAY performed better than RANDOM. Random movement can sometimes move the ship toward enemies instead of away from them. These results are specific to the current configuration and experiment.

## Game Day 13 — Training Report

### 1. Why do we use four seed groups?
Training, validation, development, and test use different seeds to avoid overlap. If we train on test seeds, the final evaluation becomes biased and is no longer a fair measure of generalization.

### 2. Difference between final_model and best_model
`final_model.zip` contains the model at the end of training. `best_model.zip` contains the model with the highest validation reward. The best model is selected using validation results, but the validation set contains only 10 games, so the result can be noisy.

### 3. Why save run_info.json?
It records the training settings, Python and package versions, machine information, network parameters, training duration, and number of steps. This makes the experiment reproducible.

### 4. What does n-envs = 8 change?
It runs eight game environments in parallel, allowing the agent to collect experience from several environments during training.

### 5. Training speed and duration
- Steps completed: 1,007,616
- Training time: 12.1 minutes
- Speed: approximately 1,386 steps/second

### 6. Validation curve and baselines
The validation reward improved during training and reached 65.8 at 1,000,000 steps. It exceeded the approximate RANDOM baseline (12) and COLUMN SHOOTER baseline (48). It remains below the approximate RULE-BASED baseline (199). The result suggests that PPO learned a useful policy, but it has not yet surpassed the rule-based agent.

### 7. Final model comparison on development seeds (0–29)

| Player | Mean reward |
|---|---:|
| STAY | -7.73 |
| STAY + SHOOT | 5.11 |
| RANDOM | 10.19 |
| COLUMN SHOOTER | 34.55 |
| PPO final_model | 70.33 |

The final PPO model outperformed the reference players on these development seeds. The best validation model obtained a mean reward of 51.96 on the same development seeds.
# Game Day 16 — Three Seeds and Official Test

## 1. What do the 3 seeds show that one run cannot show?

The three seeds show the variability of PPO training. Seed 0 achieved a validation mean of 204.1 over the last four evaluations, seed 1 achieved 185.3, and seed 2 achieved 202.4. One run alone cannot show how consistent the learning process is.

## 2. What does the grey band represent?

The grey band represents the mean validation reward plus or minus one standard deviation across the three seeds. It shows how much the results vary between training runs.

## 3. Why do we choose the main agent using the mean of the last 4 validation points, and never the test results?

A single validation evaluation uses only 10 games and can be noisy. Averaging the last four evaluations gives a more stable estimate. We use validation to select the main agent and keep the test seeds untouched for an unbiased final evaluation.

**Selected main agent: PPO (ppo_4m), seed 0.**

## 4. Explain: total_reward = +27.409 ± 2.946.

PPO (ppo_4m) achieved an average reward 27.409 points higher than the rule-based agent on the same test games, with a 95% margin of error of 2.946 points.

## 5. What does the survival result say about the two strategies?

PPO (ppo_4m) achieved a higher average reward (226.13) and more kills (130.76) than the rule-based agent (198.72 reward and 99.86 kills). However, PPO completed 49 out of 50 games, while the rule-based agent completed all 50. This suggests that PPO is more aggressive, while the rule-based strategy is more consistent in survival.

## 6. What is the rule about running test.py, and why does it matter?

We run test.py once using the frozen models and settings on the reserved test seeds (1000–1049). We must not use test results to tune or select the model because this would cause data leakage and make the final evaluation less reliable.

## 7. Official results

### OFFICIAL RESULTS — Mean over 50 test games

| Player | Total reward | Survival steps | Survived | Kills | Accuracy |
|---|---:|---:|---:|---:|---:|
| STAY | -7.31 | 766.92 | 0% | 0.00 | N/A |
| STAY + SHOOT | 7.76 | 1334.44 | 0% | 5.52 | 0.13 |
| RANDOM | 11.65 | 1344.82 | 0% | 8.88 | 0.16 |
| COLUMN SHOOTER | 48.48 | 2631.78 | 0% | 20.36 | 0.36 |
| PPO (ppo_4m_s1) | 183.95 | 4448.00 | 58% | 109.54 | 0.61 |
| PPO (ppo_4m_s2) | 190.32 | 4486.66 | 62% | 113.22 | 0.63 |
| RULE-BASED | 198.72 | 5000.00 | 100% | 99.86 | 0.56 |
| PPO (mean of 3 seeds) | 200.13 | 4641.53 | 72.67% | 117.84 | 0.64 |
| PPO (ppo_4m) | 226.13 | 4989.92 | 98% | 130.76 | 0.67 |

### Paired comparison — Each PPO seed minus RULE-BASED

| Player | Metric | Mean difference | 95% margin |
|---|---|---:|---:|
| PPO (ppo_4m) | total_reward | +27.409 | 2.946 |
| PPO (ppo_4m) | kills | +30.900 | 2.206 |
| PPO (ppo_4m) | survival_steps | -10.080 | 19.757 |
| PPO (ppo_4m) | accuracy | +0.110 | 0.008 |
| PPO (ppo_4m_s1) | total_reward | -14.770 | 14.055 |
| PPO (ppo_4m_s1) | kills | +9.680 | 7.527 |
| PPO (ppo_4m_s1) | survival_steps | -552.000 | 244.456 |
| PPO (ppo_4m_s1) | accuracy | +0.049 | 0.020 |
| PPO (ppo_4m_s2) | total_reward | -8.403 | 15.326 |
| PPO (ppo_4m_s2) | kills | +13.360 | 8.511 |
| PPO (ppo_4m_s2) | survival_steps | -513.340 | 256.828 |
| PPO (ppo_4m_s2) | accuracy | +0.066 | 0.022 |

## Generated files

- `results/learning_curves.png`
- `results/seeds_summary.csv`
- `results/test_episodes.csv`
- `results/test_summary.csv`
- `results/test_paired.csv`

## Conclusion

PPO (ppo_4m), selected using validation only, achieved the highest test reward among the evaluated agents. However, performance varies across the three PPO seeds, so reporting all three runs is important. The rule-based agent remained the most consistent in survival. The test results were obtained using the reserved test seeds without further model selection.
