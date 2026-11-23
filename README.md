# Space Survival Shooter: A PPO Agent Against a Rule-Based Agent

Project for the **Reinforcement Learning and Optimal Control 2026/2027** course (3CS AI & DS).

We developed a video game, trained a Deep Reinforcement Learning agent using Proximal Policy Optimization (PPO) with a custom neural network, designed a rule-based agent, and compared their performance.

## Group Members

- **Yasmine OUKEMOUM**
- **Alaa Errahmane CHETTAB**
- **Amina TOUATI**
- **GOUDJI ABLA**
- **MOUNTAHA**

## 1. The Game

A spaceship moves in a 600 × 600 arena and must survive as long as possible while enemies approach from the top.

- **Drifters (red):** Move straight down and are destroyed by one hit.
- **Hunters (orange):** Move toward the spaceship and require two hits to be destroyed.
- **Shield rings (green):** Block the next hit.
- **Lives:** Three lives per game.
- **Ammunition:** 20 bullets, with one bullet recharged every 20 steps.
- **Shooting cooldown:** Eight steps between shots.
- **Increasing difficulty:** Enemies appear more frequently as time passes.

The game was inspired by *Space Invaders* (Taito, 1978) and *Asteroids* (Atari, 1979).

Our contributions include limited ammunition with slow recharge, two enemy types with different behaviors, shield pickups, increasing difficulty, and four levels with different feature configurations.

The game supports two modes:

- **Interactive mode:** `play.py`, with a Pygame window.
- **Headless mode:** No graphical window; used for training and testing.

The complete game design, including the rules, 25-dimensional observation space, 10 actions, and reward function, is documented in `docs/game_design.md`.

## 2. Quick Start

### Requirements

- [uv](https://docs.astral.sh/uv/)
- Git

The project uses `.python-version`, `pyproject.toml`, and `uv.lock` to configure Python and install the required dependencies.

### Installation and execution

```bash
git clone https://github.com/TMINA2004/space-survival-shooter.git
cd space-survival-shooter
uv sync
uv run python play.py
```

### Controls

| Key | Action |
|---|---|
| Arrow keys, WASD, or ZQSD | Move |
| SPACE | Shoot |
| H | Show hitboxes |
| R | Restart |
| ESC | Quit |

## 3. Project Structure

| Path | Description |
|---|---|
| `play.py` | Play the game using the keyboard |
| `observe.py` | Watch an episode played by an RL or rule-based agent |
| `train.py` | Train PPO agents and generate models, checkpoints, and TensorBoard logs |
| `test.py` | Evaluate agents and export CSV results |
| `evaluate.ipynb` | Visualizations and comparison tables |
| `space_shooter/config.py` | Game configuration and reward weights |
| `space_shooter/game.py` | Core game mechanics |
| `space_shooter/renderer.py` | Pygame rendering |
| `space_shooter/env.py` | Gymnasium environment, observations, rewards, and episode handling |
| `space_shooter/network.py` | Custom neural network feature extractor |
| `space_shooter/callbacks.py` | Episode-level TensorBoard logging and validation |
| `space_shooter/baselines.py` | Reference agents: STAY, STAY + SHOOT, RANDOM, and COLUMN SHOOTER |
| `space_shooter/rule_agent.py` | Rule-based agent |
| `space_shooter/evaluation.py` | Episode execution and performance metrics |
| `space_shooter/rl_player.py` | Interface for loading and using trained models |
| `scripts/` | Development, testing, diagnostic, and plotting scripts |
| `docs/` | Game design and project notes |
| `results/` | Evaluation CSV files and figures |
| `final_models/` | Selected trained models and their `run_info.json` files |
| `final_logs/` | TensorBoard logs containing episode rewards |
| `models/`, `runs/` | Generated training artifacts, ignored by Git |

## 4. How to Run the Project

### Play the game

```bash
uv run python play.py
uv run python play.py 1
```

The default level is 4; levels 1–4 are available.

### Observe an agent

```bash
uv run python observe.py --agent rl
uv run python observe.py --agent rule --seed 7 --fps 120
```

Available agents: `rl`, `rule`, `column`, and `random`.

Additional options include `--model`, `--seed`, `--level`, `--fps`, `--stochastic`, and `--hitboxes`.

During observation, press P to pause, H to show hitboxes, and ESC to quit.

### Train an agent

```bash
uv run python train.py --name my_run --steps 4000000 --seed 0
```

Available options include `--name`, `--steps`, `--seed`, `--n-envs`, `--level`, `--lr`, `--gamma`, `--ent-coef`, `--network {custom,mlp}`, `--eval-every`, `--eval-episodes`, and `--checkpoint-every`.

Training outputs include TensorBoard logs, model files, checkpoints, and `run_info.json`.

### Evaluate the agents

```bash
uv run python test.py
uv run python test.py --episodes 5 --out results/quick
```

The evaluation script generates:

- `results/test_episodes.csv`
- `results/test_summary.csv`
- `results/test_paired.csv`

### View TensorBoard logs

```bash
uv run python -m tensorboard.main --logdir final_logs
```

### Run development scripts

Run scripts from the project root using Python's module syntax:

```bash
uv run python -m scripts.stress_test
uv run python -m scripts.try_game_day7
uv run python -m scripts.try_env_day8
```

### Run the evaluation notebook

Open `evaluate.ipynb` in VS Code, select the project's `.venv` Python kernel, and run all cells.

## 5. Reproducing the Results

The PPO experiments use three random seeds.

```bash
uv run python train.py --name ppo_4m    --seed 0 --steps 4000000 --eval-every 250000
uv run python train.py --name ppo_4m_s1 --seed 1 --steps 4000000 --eval-every 250000
uv run python train.py --name ppo_4m_s2 --seed 2 --steps 4000000 --eval-every 250000
uv run python -m scripts.plot_seeds
uv run python test.py
```

A training run takes approximately 45 minutes on a laptop CPU, according to the project template. No GPU is required.

The project separates random seeds by purpose:

- **Training:** Seeds starting from 10,000.
- **Validation:** Seeds 2000–2009, used for model selection.
- **Development:** Seeds 0–29, used for debugging.
- **Testing:** Seeds 1000–1049, used once for final evaluation.

## 6. Results

The following results are reported for 50 test games using seeds 1000–1049.

| Agent | Mean reward | Kills | Accuracy | Games survived |
|---|---:|---:|---:|---:|
| RANDOM | 11.65 | 8.9 | 16% | 0% |
| COLUMN SHOOTER | 48.48 | 20.4 | 36% | 0% |
| RULE-BASED | 198.72 | 99.9 | 56% | 100% |
| PPO (seed 0) | 226.13 | 130.8 | 67% | 98% |
| PPO (seed 1) | 183.95 | 109.5 | 61% | 58% |
| PPO (seed 2) | 190.32 | 113.2 | 63% | 62% |
| **PPO (mean of 3 seeds)** | **200.13** | **117.8** | **64%** | **73%** |

The PPO agent achieves a mean reward comparable to the rule-based agent, eliminates more enemies, and has higher shooting accuracy. However, its survival rate varies considerably across seeds and is lower on average.

See `evaluate.ipynb` for the detailed plots and comparisons.


## 7. References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR, 22(268), 1–8.
- Towers, M. et al. (2024). *Gymnasium: A Standard Interface for Reinforcement Learning Environments*. arXiv:2407.17032.
- Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd edition. MIT Press.
- [Pygame documentation](https://www.pygame.org/docs/).
- Inspirations: *Space Invaders* (Taito, 1978) and *Asteroids* (Atari, 1979).
