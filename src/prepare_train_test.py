from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("ACDT PROJECT - PREPARING TRAIN AND TEST DATA")
print("=" * 60)

# --------------------------------------------------
# 1. Locate feature dataset
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "features.csv"
)

if not input_path.exists():
    raise FileNotFoundError(
        f"Feature dataset not found at: {input_path}"
    )

print("\n[OK] Feature dataset found.")
print(f"Input file: {input_path}")

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

data = pd.read_csv(input_path)

print("\n[OK] Dataset loaded.")
print(f"Original shape: {data.shape}")

# --------------------------------------------------
# 3. Remove first interaction of each student
# --------------------------------------------------

before_rows = len(data)

data = data[
    data["previous_interactions"] > 0
].copy()

after_rows = len(data)

print("\nRemoved first interaction of each student:")
print(f"Rows removed: {before_rows - after_rows}")
print(f"Rows remaining: {after_rows}")

# --------------------------------------------------
# 4. Define input features and target
# --------------------------------------------------

feature_columns = [
    "attempt_count",
    "hint_count",
    "response_time_seconds",
    "previous_interactions",
    "previous_correct",
    "previous_incorrect",
    "previous_accuracy"
]

target_column = "correct"

X = data[feature_columns].copy()
y = data[target_column].copy()

# Replace missing response times with the median
X["response_time_seconds"] = (
    X["response_time_seconds"]
    .fillna(X["response_time_seconds"].median())
)

# --------------------------------------------------
# 5. Split by student
# --------------------------------------------------

student_ids = data["user_id"].unique()

train_students, test_students = train_test_split(
    student_ids,
    test_size=0.20,
    random_state=42
)

train_data = data[
    data["user_id"].isin(train_students)
].copy()

test_data = data[
    data["user_id"].isin(test_students)
].copy()

X_train = train_data[feature_columns].copy()
y_train = train_data[target_column].copy()

X_test = test_data[feature_columns].copy()
y_test = test_data[target_column].copy()

# Apply the training median to both datasets
training_median = (
    X_train["response_time_seconds"].median()
)

X_train["response_time_seconds"] = (
    X_train["response_time_seconds"]
    .fillna(training_median)
)

X_test["response_time_seconds"] = (
    X_test["response_time_seconds"]
    .fillna(training_median)
)

# --------------------------------------------------
# 6. Save train and test files
# --------------------------------------------------

train_path = (
    project_root
    / "data"
    / "processed"
    / "train.csv"
)

test_path = (
    project_root
    / "data"
    / "processed"
    / "test.csv"
)

X_train.assign(correct=y_train).to_csv(
    train_path,
    index=False
)

X_test.assign(correct=y_test).to_csv(
    test_path,
    index=False
)

# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAIN/TEST PREPARATION COMPLETED")
print("=" * 60)

print(f"Training students: {len(train_students)}")
print(f"Testing students : {len(test_students)}")

print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")

print(f"\nTraining file: {train_path}")
print(f"Testing file : {test_path}")

print("\nTraining correctness rate:")
print(f"{y_train.mean() * 100:.2f}%")

print("\nTesting correctness rate:")
print(f"{y_test.mean() * 100:.2f}%")

print("\nFeatures used:")
for column in feature_columns:
    print(f"- {column}")

print("\n[OK] Train/test preparation completed successfully.")