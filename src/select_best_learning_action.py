from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - SELECTING BEST LEARNING ACTION")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root
    / "models"
    / "action_transition_model.csv"
)

output_path = (
    project_root
    / "results"
    / "recommended_learning_actions.csv"
)

action_model = pd.read_csv(model_path)

print("\n[OK] Action-conditioned model loaded.")

# Define the value of each learner state.
# Higher value means a better learner condition.
state_values = {
    "initial_state": 0.50,
    "stable_learning": 1.00,
    "moderate_struggle": 0.50,
    "high_struggle": 0.00
}

action_model["next_state_value"] = (
    action_model["next_learner_state"]
    .map(state_values)
)

action_model["weighted_value"] = (
    action_model["adjusted_probability"]
    * action_model["next_state_value"]
)

action_scores = (
    action_model
    .groupby(
        ["learner_state", "action"],
        as_index=False
    )["weighted_value"]
    .sum()
)

action_scores = action_scores.rename(
    columns={
        "weighted_value": "expected_future_value"
    }
)

best_actions = (
    action_scores
    .sort_values(
        [
            "learner_state",
            "expected_future_value"
        ],
        ascending=[True, False]
    )
    .groupby("learner_state")
    .head(1)
    .reset_index(drop=True)
)

best_actions = best_actions.rename(
    columns={
        "action": "recommended_action"
    }
)

best_actions.to_csv(
    output_path,
    index=False
)

print("\n[OK] Best learning actions selected.")
print(f"Saved to: {output_path}")

print("\nExpected value of every action:")
print(
    action_scores
    .sort_values(
        ["learner_state", "expected_future_value"],
        ascending=[True, False]
    )
    .to_string(index=False)
)

print("\nRecommended action for each learner state:")
print(
    best_actions.to_string(index=False)
)

print("\n[IMPORTANT]")
print(
    "Recommendations are based on hypothetical prototype "
    "transition effects."
)
print(
    "They are not experimentally validated interventions."
)

print("\n[OK] Action selection completed.")