from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("ACDT PROJECT - VISUALIZING RECOMMENDATIONS")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "results"
    / "learner_recommendations.csv"
)

output_path = (
    project_root
    / "results"
    / "recommendation_distribution.png"
)

data = pd.read_csv(input_path)

recommendation_counts = (
    data["recommended_action"]
    .value_counts()
)

print("\nRecommendation counts:")
print(recommendation_counts)

plt.figure(figsize=(9, 5))

recommendation_counts.plot(
    kind="bar"
)

plt.title("Distribution of Personalized Learning Recommendations")
plt.xlabel("Recommended Learning Action")
plt.ylabel("Number of Learners")

plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(output_path, dpi=300)
plt.close()

print("\n[OK] Recommendation chart created.")
print(f"Saved to: {output_path}")

print("\n[OK] Visualization completed.")