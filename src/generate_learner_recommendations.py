from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - GENERATING LEARNER RECOMMENDATIONS")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

state_path = (
    project_root
    / "data"
    / "processed"
    / "learner_state.csv"
)

recommendation_path = (
    project_root
    / "results"
    / "personalized_recommendations.csv"
)

output_path = (
    project_root
    / "results"
    / "learner_recommendations.csv"
)

learner_state_data = pd.read_csv(state_path)
recommendations = pd.read_csv(recommendation_path)

print("\n[OK] Learner-state data loaded.")
print("[OK] Recommendation rules loaded.")

# Select each learner's latest observed state.
latest_learner_state = (
    learner_state_data
    .sort_values(["user_id", "order_id"])
    .groupby("user_id")
    .tail(1)
)

latest_learner_state = latest_learner_state[
    [
        "user_id",
        "order_id",
        "learner_state",
        "knowledge_state",
        "struggle_score",
        "engagement_difficulty_score"
    ]
]

learner_recommendations = latest_learner_state.merge(
    recommendations,
    on="learner_state",
    how="left"
)

learner_recommendations.to_csv(
    output_path,
    index=False
)

print("\n[OK] Learner-level recommendations generated.")
print(f"Saved to: {output_path}")

print("\nReport shape:")
print(learner_recommendations.shape)

print("\nRecommendation distribution:")
print(
    learner_recommendations[
        "recommended_action"
    ].value_counts()
)

print("\nSample learner recommendations:")
print(
    learner_recommendations
    .head(15)
    .to_string(index=False)
)

print("\n[OK] Learner recommendation generation completed.")