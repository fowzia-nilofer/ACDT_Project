from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - DATASET ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 1. Locate the cleaned dataset
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

clean_path = (
    project_root
    / "data"
    / "processed"
    / "clean_interactions.csv"
)

if not clean_path.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found at: {clean_path}"
    )

print("\n[OK] Cleaned dataset found.")
print(f"Input file: {clean_path}")

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

data = pd.read_csv(clean_path)

print("\n[OK] Dataset loaded.")
print(f"Dataset shape: {data.shape}")

# --------------------------------------------------
# 3. Check missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = data.isnull().sum()

missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    print("No missing values found.")
else:
    print(missing_values)

# --------------------------------------------------
# 4. Check unique values in important columns
# --------------------------------------------------

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

print(f"Unique students : {data['user_id'].nunique()}")
print(f"Unique problems  : {data['problem_id'].nunique()}")
print(f"Unique skills    : {data['skill_id'].nunique()}")
print(f"Unique sequences : {data['sequence_id'].nunique()}")

# --------------------------------------------------
# 5. Analyze interactions per student
# --------------------------------------------------

print("\n" + "=" * 60)
print("INTERACTIONS PER STUDENT")
print("=" * 60)

interactions_per_student = data.groupby("user_id").size()

print(
    f"Minimum interactions : "
    f"{interactions_per_student.min()}"
)

print(
    f"Maximum interactions : "
    f"{interactions_per_student.max()}"
)

print(
    f"Average interactions : "
    f"{interactions_per_student.mean():.2f}"
)

print(
    f"Median interactions  : "
    f"{interactions_per_student.median():.0f}"
)

# --------------------------------------------------
# 6. Analyze correctness by skill
# --------------------------------------------------

print("\n" + "=" * 60)
print("CORRECTNESS BY SKILL")
print("=" * 60)

skill_summary = (
    data.groupby(["skill_id", "skill_name"])["correct"]
    .agg(
        total_attempts="count",
        average_correctness="mean"
    )
    .reset_index()
)

skill_summary["average_correctness"] = (
    skill_summary["average_correctness"] * 100
)

skill_summary = skill_summary.sort_values(
    by="total_attempts",
    ascending=False
)

print("\nTop 10 most frequently practiced skills:")

print(
    skill_summary.head(10).to_string(index=False)
)

# --------------------------------------------------
# 7. Save skill summary
# --------------------------------------------------

summary_path = (
    project_root
    / "data"
    / "processed"
    / "skill_summary.csv"
)

skill_summary.to_csv(summary_path, index=False)

print("\n[OK] Skill summary saved.")
print(f"Saved file: {summary_path}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)