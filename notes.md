# Day 1 — Game Design Questions

## 1. How could the agent cheat the reward system?

The agent could stay in a safe corner to collect survival rewards without fighting enemies. To prevent this, the game can gradually increase the difficulty and give the agent stronger reasons to avoid enemies and survive actively.

## 2. Why use relative positions instead of absolute coordinates?

Relative positions tell the agent where enemies are in relation to the spaceship. This helps the agent learn how to react to nearby threats, even when the spaceship and enemies are in different parts of the screen.

## 3. What is the difference between termination and truncation?

Termination happens when the spaceship loses all its lives. Truncation happens when the episode reaches the maximum limit of 5,000 steps, even if the spaceship is still alive.

## 4. How could we simplify the observation space?

We could remove information about the fourth-nearest enemy. This would reduce the number of observations, but the agent would have less information about distant threats and might make worse decisions.