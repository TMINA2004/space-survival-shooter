import glob
import os

import matplotlib

matplotlib.use("Agg")                      # draw to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from space_shooter.baselines import column_shooter
from space_shooter.evaluation import evaluate_player
from space_shooter.rule_agent import RuleBasedAgent

RUNS = {"seed 0": "ppo_4m", "seed 1": "ppo_4m_s1", "seed 2": "ppo_4m_s2"}
VALIDATION_SEEDS = range(2000, 2010)


def find_log_dir(name):
    # "ppo_4m_[0-9]*" matches ppo_4m_1 but NOT ppo_4m_s1_1
    folders = sorted(glob.glob(os.path.join("runs", f"{name}_[0-9]*")))
    if not folders:
        raise SystemExit(f"No TensorBoard folder found for run '{name}' in runs/")
    return folders[-1]


def read_scalar(folder, tag):
    accumulator = EventAccumulator(folder, size_guidance={"scalars": 0})
    accumulator.Reload()
    return {event.step: event.value for event in accumulator.Scalars(tag)}


validation, training = {}, {}
for label, name in RUNS.items():
    folder = find_log_dir(name)
    validation[label] = read_scalar(folder, "eval/mean_reward")
    training[label] = read_scalar(folder, "rollout/ep_rew_mean")
    print(f"{label}: {folder}")

print("\nMeasuring the reference players on the same validation games (about 1 minute)...")
references = {"RULE-BASED": RuleBasedAgent(), "COLUMN SHOOTER": column_shooter}
reference_reward = {
    name: evaluate_player(name, player, VALIDATION_SEEDS)["total_reward"].mean()
    for name, player in references.items()
}

# Evaluation steps that exist in all three runs
steps = sorted(set.intersection(*[set(v) for v in validation.values()]))
matrix = np.array([[validation[label][s] for s in steps] for label in RUNS])    # seeds x steps
mean, std = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for label in RUNS:
    xs = sorted(training[label])
    ax1.plot(xs, [training[label][x] for x in xs], label=label)
ax1.set_title("Training reward (mean of the last 100 episodes)")
ax1.set_xlabel("training steps")
ax1.set_ylabel("total reward per episode")
ax1.grid(alpha=0.3)
ax1.legend()

for i, label in enumerate(RUNS):
    ax2.plot(steps, matrix[i], alpha=0.35, label=label)
ax2.plot(steps, mean, color="black", linewidth=2.5, label="mean of 3 seeds")
ax2.fill_between(steps, mean - std, mean + std, color="black", alpha=0.12)
for name, value in reference_reward.items():
    ax2.axhline(value, linestyle="--", linewidth=1, label=f"{name} ({value:.0f})")
ax2.set_title("Validation reward (10 fixed games, seeds 2000-2009)")
ax2.set_xlabel("training steps")
ax2.set_ylabel("total reward per episode")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8)

fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/learning_curves.png", dpi=150)

rows = []
for i, label in enumerate(RUNS):
    rows.append({"run": label, "final": matrix[i][-1],
                 "mean_last_4": matrix[i][-4:].mean(), "best": matrix[i].max()})
table = pd.DataFrame(rows).round(1)
table.to_csv("results/seeds_summary.csv", index=False)

print("\nValidation reward per run:")
print(table.to_string(index=False))
print(f"\nAt the last evaluation: mean {mean[-1]:.1f}, std between seeds {std[-1]:.1f}")
chosen = table.loc[table["mean_last_4"].idxmax(), "run"]
print(f"Main agent, chosen by VALIDATION only (best mean of the last 4 evaluations): {chosen}")
print("Figure saved: results/learning_curves.png")