from pathlib import Path
import pandas as pd


print("=" * 60)
print("ACDT PROJECT - INSPECTING ACTION EFFECTS")
print("=" * 60)


project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "models"
    / "action_transition_model.csv"
)

data = pd.read_csv(input_path)

print("\nAvailable columns:")
print(list(data.columns))

print("\nAction-conditioned transition model:")
print(data.to_string(index=False))

print("\nAverage probability of reaching high_struggle by action:")

high_struggle_data = data[
    data["next_learner_state"] == "high_struggle"
]

action_summary = (
    high_struggle_data
    .groupby("action")["adjusted_probability"]
    .mean()
    .sort_values()
)

print(action_summary.to_string())

print("\n[OK] Action effects inspection completed.")