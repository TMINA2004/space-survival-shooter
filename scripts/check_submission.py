import json
import os
import tomllib

REQUIRED_FILES = ["train.py", "test.py", "play.py", "observe.py", "evaluate.ipynb", "README.md",
                  "pyproject.toml", "uv.lock", ".python-version", "docs/game_design.md"]
REQUIRED_RESULTS = ["results/test_episodes.csv", "results/test_summary.csv", "results/test_paired.csv",
                    "results/seeds_summary.csv", "results/learning_curves.png"]
MODELS = ["final_models/ppo_4m/final_model.zip", "final_models/ppo_4m_s1/final_model.zip",
          "final_models/ppo_4m_s2/final_model.zip"]
LIBRARIES = ["gymnasium", "pygame", "numpy", "torch", "stable-baselines3", "tensorboard", "pandas", "matplotlib"]

problems = []


def check(condition, message):
    print(("PASS  " if condition else "FAIL  ") + message)
    if not condition:
        problems.append(message)


for path in REQUIRED_FILES + REQUIRED_RESULTS + MODELS:
    check(os.path.exists(path), f"exists: {path}")

for path in MODELS:
    if os.path.exists(path):
        check(os.path.getsize(path) < 5_000_000, f"small enough: {path}")

check(os.path.isdir("final_logs") and len(os.listdir("final_logs")) > 0, "final_logs/ contains the TensorBoard logs")

if os.path.exists("evaluate.ipynb"):
    with open("evaluate.ipynb", encoding="utf-8") as f:
        notebook = json.load(f)
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    with_outputs = [c for c in code_cells if c.get("outputs")]
    check(len(with_outputs) == len(code_cells),
          f"notebook has saved outputs in every code cell ({len(with_outputs)}/{len(code_cells)})")

if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        text = f.read()
    check("<<FILL" not in text, "README has no <<FILL ...>> placeholder left")
    check("uv sync" in text, "README explains `uv sync`")

if os.path.exists("pyproject.toml"):
    with open("pyproject.toml", "rb") as f:
        dependencies = " ".join(tomllib.load(f)["project"].get("dependencies", [])).lower()
    for name in LIBRARIES:
        check(name in dependencies, f"pyproject.toml lists {name}")

print()
print("READY: every check passed." if not problems else f"{len(problems)} problem(s) to fix.")