from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - CREATING MODEL FEATURES")
print("=" * 60)

# --------------------------------------------------
# 1. Locate modeling dataset
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "modeling_interactions.csv"
)

if not input_path.exists():
    raise FileNotFoundError(
        f"Modeling dataset not found at: {input_path}"
    )

print("\n[OK] Modeling dataset found.")
print(f"Input file: {input_path}")

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

data = pd.read_csv(input_path)

print("\n[OK] Dataset loaded.")
print(f"Original shape: {data.shape}")

# --------------------------------------------------
# 3. Sort by student and interaction order
# --------------------------------------------------

data = data.sort_values(
    by=["user_id", "order_id"]
).reset_index(drop=True)

# --------------------------------------------------
# 4. Create previous-performance features
# --------------------------------------------------

print("\nCreating student history features...")

# Number of previous interactions by this student
data["previous_interactions"] = (
    data.groupby("user_id").cumcount()
)

# Number of previous correct answers
data["previous_correct"] = (
    data.groupby("user_id")["correct"]
    .cumsum()
    - data["correct"]
)

# Previous incorrect answers
data["previous_incorrect"] = (
    data["previous_interactions"]
    - data["previous_correct"]
)

# Previous accuracy
data["previous_accuracy"] = 0.0

has_history = data["previous_interactions"] > 0

data.loc[has_history, "previous_accuracy"] = (
    data.loc[has_history, "previous_correct"]
    / data.loc[has_history, "previous_interactions"]
)

# --------------------------------------------------
# 5. Create response-time and hint features
# --------------------------------------------------

data["response_time_seconds"] = pd.to_numeric(
    data["response_time_seconds"],
    errors="coerce"
)

data["hint_count"] = pd.to_numeric(
    data["hint_count"],
    errors="coerce"
).fillna(0)

data["attempt_count"] = pd.to_numeric(
    data["attempt_count"],
    errors="coerce"
).fillna(0)

# --------------------------------------------------
# 6. Select useful modeling columns
# --------------------------------------------------

feature_columns = [
    "user_id",
    "problem_id",
    "skill_id",
    "skill_name",
    "order_id",
    "correct",
    "attempt_count",
    "hint_count",
    "response_time_seconds",
    "previous_interactions",
    "previous_correct",
    "previous_incorrect",
    "previous_accuracy"
]

features = data[feature_columns].copy()

# --------------------------------------------------
# 7. Save feature dataset
# --------------------------------------------------

output_path = (
    project_root
    / "data"
    / "processed"
    / "features.csv"
)

features.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("FEATURE CREATION COMPLETED")
print("=" * 60)

print(f"Feature dataset shape: {features.shape}")
print(f"Saved file: {output_path}")

print("\nCreated features:")
print("- previous_interactions")
print("- previous_correct")
print("- previous_incorrect")
print("- previous_accuracy")
print("- response_time_seconds")
print("- hint_count")
print("- attempt_count")

print("\nFirst five rows of selected features:")
print(features.head().to_string(index=False))

print("\n[OK] Feature creation completed successfully.")