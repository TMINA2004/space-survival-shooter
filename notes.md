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