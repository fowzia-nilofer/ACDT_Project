from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - VALIDATING ACTION-CONDITIONED MODEL")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root
    / "models"
    / "action_transition_model.csv"
)

action_model = pd.read_csv(model_path)

print("\n[OK] Action-conditioned model loaded.")
print(f"Model shape: {action_model.shape}")

print("\nChecking probability sums...")

probability_sums = (
    action_model
    .groupby(
        ["learner_state", "action"]
    )["adjusted_probability"]
    .sum()
)

print(probability_sums.to_string())

invalid_groups = probability_sums[
    (probability_sums < 0.99)
    | (probability_sums > 1.01)
]

if len(invalid_groups) == 0:
    print("\n[OK] All state-action probability sums are valid.")
else:
    print("\n[WARNING] Some probability sums are invalid:")
    print(invalid_groups)

print("\nChecking available actions...")

print(
    action_model["action"]
    .unique()
)

print("\nChecking learner states...")

print(
    action_model["learner_state"]
    .unique()
)

print("\n[OK] Action-conditioned model validation completed.")