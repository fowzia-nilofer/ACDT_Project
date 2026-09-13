from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


print("=" * 60)
print("ACDT PROJECT - VISUALIZING STATE RECOMMENDATIONS")
print("=" * 60)


project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "results"
    / "state_recommendation_summary.csv"
)

output_path = (
    project_root
    / "results"
    / "state_recommendation_summary.png"
)


data = pd.read_csv(input_path)

labels = (
    data["learner_state"]
    + " → "
    + data["recommended_action"]
)

values = data["learner_count"]


plt.figure(figsize=(10, 6))

plt.bar(labels, values)

plt.title("Learner State and Recommended Learning Action")
plt.xlabel("Learner State → Recommended Action")
plt.ylabel("Number of Learners")

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(output_path, dpi=300)

plt.show()

print("\n[OK] Visualization created.")
print(f"Saved to: {output_path}")

print("\n[OK] Visualization completed.")