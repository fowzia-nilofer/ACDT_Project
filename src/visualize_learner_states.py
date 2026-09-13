from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


print("=" * 60)
print("ACDT PROJECT - VISUALIZING LEARNER STATES")
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
    / "learner_state_distribution.png"
)


data = pd.read_csv(input_path)

data = (
    data
    .groupby("learner_state")["learner_count"]
    .sum()
    .reset_index()
)

data = data.sort_values(
    "learner_count",
    ascending=False
)


plt.figure(figsize=(9, 6))

bars = plt.bar(
    data["learner_state"],
    data["learner_count"]
)

plt.title("Distribution of Learner States")
plt.xlabel("Learner State")
plt.ylabel("Number of Learners")

plt.xticks(rotation=20)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300
)

plt.show()

print("\n[OK] Learner-state visualization created.")
print(f"Saved to: {output_path}")

print("\n[OK] Learner-state visualization completed.")