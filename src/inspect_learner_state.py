from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - INSPECTING LEARNER STATES")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "learner_state.csv"
)

data = pd.read_csv(input_path)

print("\n[OK] Learner-state dataset loaded.")
print(f"Dataset shape: {data.shape}")

print("\nFirst 10 learner-state records:")
print(data.head(10).to_string(index=False))

print("\nHighest-struggle records:")
highest_struggle = data.sort_values(
    by="struggle_score",
    ascending=False
)

print(
    highest_struggle[
        [
            "user_id",
            "order_id",
            "correct",
            "knowledge_state",
            "struggle_score",
            "engagement_difficulty_score",
            "learner_state"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nLowest-knowledge-state records:")
lowest_knowledge = data.sort_values(
    by="knowledge_state",
    ascending=True
)

print(
    lowest_knowledge[
        [
            "user_id",
            "order_id",
            "correct",
            "knowledge_state",
            "struggle_score",
            "engagement_difficulty_score",
            "learner_state"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nAverage scores by learner state:")

summary = (
    data
    .groupby("learner_state")[
        [
            "knowledge_state",
            "struggle_score",
            "engagement_difficulty_score"
        ]
    ]
    .mean()
    .round(4)
)

print(summary)

print("\n[OK] Learner-state inspection completed.")