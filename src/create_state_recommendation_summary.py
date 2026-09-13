from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - CREATING STATE RECOMMENDATION SUMMARY")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

learner_path = (
    project_root
    / "results"
    / "learner_recommendations.csv"
)

output_path = (
    project_root
    / "results"
    / "state_recommendation_summary.csv"
)

data = pd.read_csv(learner_path)

state_summary = (
    data
    .groupby(
        [
            "learner_state",
            "recommended_action"
        ]
    )
    .size()
    .reset_index(name="learner_count")
)

total_learners = state_summary["learner_count"].sum()

state_summary["percentage"] = (
    state_summary["learner_count"]
    / total_learners
    * 100
).round(2)

state_summary = state_summary.sort_values(
    "learner_count",
    ascending=False
)

state_summary.to_csv(
    output_path,
    index=False
)

print("\n[OK] State-recommendation summary created.")
print(f"Saved to: {output_path}")

print("\nSummary:")
print(
    state_summary.to_string(index=False)
)

print("\n[OK] State-recommendation summary completed.")