from pathlib import Path
import pandas as pd


print("=" * 60)
print("ACDT PROJECT - CREATING LEARNER TRAJECTORIES")
print("=" * 60)


project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "learner_state.csv"
)

output_path = (
    project_root
    / "data"
    / "processed"
    / "learner_trajectories.csv"
)


data = pd.read_csv(input_path)

print("\n[OK] Learner-state dataset loaded.")
print(f"Dataset shape: {data.shape}")


# Sort each learner's records chronologically
data = data.sort_values(
    by=["user_id", "order_id"]
).reset_index(drop=True)


# Create the next learner state
data["next_learner_state"] = (
    data
    .groupby("user_id")["learner_state"]
    .shift(-1)
)


# Create the next knowledge state
data["next_knowledge_state"] = (
    data
    .groupby("user_id")["knowledge_state"]
    .shift(-1)
)


# Create the next struggle score
data["next_struggle_score"] = (
    data
    .groupby("user_id")["struggle_score"]
    .shift(-1)
)


# Mark the final interaction of each learner
data["is_final_interaction"] = (
    data["next_learner_state"].isna()
)


output_columns = [
    "user_id",
    "order_id",
    "problem_id",
    "correct",
    "knowledge_state",
    "struggle_score",
    "engagement_difficulty_score",
    "learner_state",
    "next_learner_state",
    "next_knowledge_state",
    "next_struggle_score",
    "is_final_interaction"
]

output = data[output_columns]


output.to_csv(
    output_path,
    index=False
)


print("\n[OK] Learner trajectories created.")
print(f"Saved to: {output_path}")
print(f"Output shape: {output.shape}")


print("\nLearner-state transition counts:")

transitions = (
    output[
        ~output["is_final_interaction"]
    ]
    .groupby(
        [
            "learner_state",
            "next_learner_state"
        ]
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print(transitions.head(20))


print("\nNumber of final interactions:")
print(
    output["is_final_interaction"].sum()
)


print("\n[OK] Learner-trajectory creation completed.")