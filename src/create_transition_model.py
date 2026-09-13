from pathlib import Path
import pandas as pd


print("=" * 60)
print("ACDT PROJECT - CREATING TRANSITION MODEL")
print("=" * 60)


project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "learner_trajectories.csv"
)

output_path = (
    project_root
    / "models"
    / "transition_probability_model.csv"
)


data = pd.read_csv(input_path)

print("\n[OK] Learner trajectories loaded.")
print(f"Dataset shape: {data.shape}")


# Remove final interactions because they have no next state
transitions = data[
    data["is_final_interaction"] == False
].copy()


# Count each state transition
transition_counts = (
    transitions
    .groupby(
        [
            "learner_state",
            "next_learner_state"
        ]
    )
    .size()
    .reset_index(
        name="transition_count"
    )
)


# Calculate probabilities within each current state
transition_counts["transition_probability"] = (
    transition_counts["transition_count"]
    / transition_counts
        .groupby("learner_state")["transition_count"]
        .transform("sum")
)


# Round probabilities for readability
transition_counts["transition_probability"] = (
    transition_counts["transition_probability"]
    .round(4)
)


transition_counts.to_csv(
    output_path,
    index=False
)


print("\n[OK] Transition probability model created.")
print(f"Saved to: {output_path}")


print("\nTransition probability table:")
print(
    transition_counts
    .sort_values(
        [
            "learner_state",
            "transition_probability"
        ],
        ascending=[True, False]
    )
    .to_string(index=False)
)


print("\n[OK] Transition-model creation completed.")