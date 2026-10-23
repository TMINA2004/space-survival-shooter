# Space Survival Shooter: Game Design

## 1. Game description

The player controls a spaceship in a square arena. Enemies enter from the top of the arena.

The player's objective is to survive as long as possible and destroy enemies.

The game ends when the player loses all three lives. The difficulty increases over time.

The game will be developed in Python. Pygame will be used for graphics, and Gymnasium will be used to create the reinforcement learning environment.

## 2. Original contributions

- Limited ammunition with automatic recharge.
- Two enemy types: Drifter and Hunter.
- A shield power-up that absorbs one hit.
- Difficulty increases over time.
- Four development levels to evaluate the learning agent.

## 3. Arena and objects

| Object | Size (radius) | Speed per step | Behavior |
|---|---:|---:|---|
| Arena | 600 x 600 | - | The ship cannot leave the arena |
| Ship | 15 | 6 | Moves in four directions |
| Bullet | 4 | 12 | Travels upward |
| Drifter | 15 | 3 | Moves downward and needs one hit to be destroyed |
| Hunter | 15 | 1.5 | Chases the ship and needs two hits to be destroyed |
| Shield pickup | 10 | 2 | Falls downward and gives the ship a shield |

## 4. Game rules

- The player starts with 3 lives and 20 ammunition units.
- The ship starts near the bottom center of the arena.
- The player can move up, down, left, and right.
- When an enemy touches the ship, the ship loses one life unless its shield absorbs the hit.
- An enemy that hits the ship is destroyed.
- After taking a hit, the ship is invulnerable for 60 steps.
- Shooting costs one ammunition unit.
- The shooting cooldown is 8 steps.
- Maximum ammunition is 20 units.
- One ammunition unit is restored every 20 steps.
- Enemies spawn more frequently as time passes.
- In Level 4, each new enemy has a 25% probability of being a Hunter.
- Shield pickups appear approximately every 400 steps.
- Enemies that leave the bottom of the arena disappear without a penalty.
- The game ends when the player has zero lives.
- During training, an episode is truncated after 5000 steps as a technical safety limit.

## 5. Development levels

| Level | Features |
|---|---|
| 1 | Movement and Drifter enemies; the agent learns to dodge |
| 2 | Shooting with unlimited ammunition |
| 3 | Limited ammunition and automatic recharge |
| 4 | Hunter enemies and shield pickups |

## 6. Action space

The agent has six possible actions, represented by Discrete(6).

| Action | Meaning |
|---:|---|
| 0 | Stay still |
| 1 | Move up |
| 2 | Move down |
| 3 | Move left |
| 4 | Move right |
| 5 | Shoot |

## 7. Observation space

The agent receives 25 numerical values describing the current game situation.

| Index | Meaning |
|---|---|
| 0-1 | Ship x and y coordinates, normalized between 0 and 1 |
| 2 | Remaining lives divided by 3 |
| 3 | Remaining ammunition divided by 20 |
| 4 | Shield status: 0 or 1 |
| 5 | Shooting cooldown divided by 8 |
| 6-9 | Relative position, type, and existence of the nearest enemy |
| 10-13 | Relative position, type, and existence of the second-nearest enemy |
| 14-17 | Relative position, type, and existence of the third-nearest enemy |
| 18-21 | Relative position, type, and existence of the fourth-nearest enemy |
| 22-24 | Relative position and existence of the nearest shield pickup |

Relative positions are scaled to a small numerical range. Missing objects are represented by zeros.

## 8. Reward function

These are initial reward values and may be adjusted after testing.

| Event | Reward |
|---|---:|
| Each step survived | +0.01 |
| Destroy a Drifter | +1 |
| Destroy a Hunter | +2 |
| Lose a life | -5 |
| Collect a shield | +0.5 |

The reward function encourages survival and destroying enemies while discouraging the loss of lives.

## 9. Episode ending

- Termination: the player loses all lives.
- Truncation: the training episode reaches 5000 steps.

The step limit is a technical safety measure for training. It does not introduce a time limit into the normal game rules.

## 10. Evaluation metrics

The environment will record:

- Total reward.
- Number of steps survived.
- Game score.
- Number of Drifters destroyed.
- Number of Hunters destroyed.
- Number of shots fired.
- Number of successful hits.
- Shooting accuracy.
- Number of lives lost.
- Number of shields collected.

## 11. Planned project structure

- `config.py`: game settings and numerical parameters.
- `game.py`: game logic without graphics.
- `renderer.py`: graphical display using Pygame.
- `env.py`: reinforcement learning environment using Gymnasium.
- `rule_agent.py`: a simple rule-based agent.
- `docs/game_design.md`: game design document.
- `notes.md`: explanations and answers to the design questions.