from pathlib import Path
import pandas as pd


print("=" * 60)
print("ACDT PROJECT - CLEANING DATASET")
print("=" * 60)

# --------------------------------------------------
# 1. Locate project folders and files
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

raw_path = (
    project_root
    / "data"
    / "raw"
    / "assistments_skill_builder.csv"
)

processed_folder = (
    project_root
    / "data"
    / "processed"
)

processed_folder.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    processed_folder
    / "clean_interactions.csv"
)

# --------------------------------------------------
# 2. Check whether the raw dataset exists
# --------------------------------------------------

if not raw_path.exists():
    print("\nERROR: Raw dataset was not found.")
    print(raw_path)
    raise SystemExit

print("\n[OK] Raw dataset found.")
print(f"Input file: {raw_path}")

# --------------------------------------------------
# 3. Select only useful columns
# --------------------------------------------------

selected_columns = [
    "order_id",
    "sequence_id",
    "user_id",
    "problem_id",
    "skill_id",
    "skill_name",
    "correct",
    "attempt_count",
    "hint_count",
    "hint_total",
    "ms_first_response",
    "overlap_time",
    "opportunity",
    "opportunity_original",
]

print("\nReading required columns...")

data = pd.read_csv(
    raw_path,
    usecols=selected_columns,
    encoding="latin1",
    low_memory=False
)

print(f"Original selected shape: {data.shape}")

# --------------------------------------------------
# 4. Remove completely empty rows
# --------------------------------------------------

data = data.dropna(
    how="all"
)

# --------------------------------------------------
# 5. Convert numeric columns
# --------------------------------------------------

numeric_columns = [
    "order_id",
    "sequence_id",
    "user_id",
    "problem_id",
    "correct",
    "attempt_count",
    "hint_count",
    "hint_total",
    "ms_first_response",
    "overlap_time",
    "opportunity",
    "opportunity_original",
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

# --------------------------------------------------
# 6. Remove rows without essential information
# --------------------------------------------------

essential_columns = [
    "user_id",
    "problem_id",
    "correct",
]

before_removal = len(data)

data = data.dropna(
    subset=essential_columns
)

after_removal = len(data)

print(
    f"\nRemoved rows with missing essential values: "
    f"{before_removal - after_removal:,}"
)

# --------------------------------------------------
# 7. Sort student interactions
# --------------------------------------------------

data = data.sort_values(
    by=[
        "user_id",
        "order_id",
    ],
    kind="stable"
)

# --------------------------------------------------
# 8. Reset row numbers
# --------------------------------------------------

data = data.reset_index(
    drop=True
)

# --------------------------------------------------
# 9. Save cleaned dataset
# --------------------------------------------------

data.to_csv(
    output_path,
    index=False
)

# --------------------------------------------------
# 10. Display summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print(f"\nCleaned dataset shape: {data.shape}")
print(f"Saved file: {output_path}")

print("\nCleaned columns:")
for column in data.columns:
    print(f"- {column}")

print("\nFirst five cleaned rows:")
print(
    data.head().to_string(index=False)
)

print("\nNumber of unique students:")
print(data["user_id"].nunique())

print("\nNumber of unique problems:")
print(data["problem_id"].nunique())

print("\nNumber of unique skills:")
print(data["skill_id"].nunique())

print("\nCorrectness distribution:")
print(data["correct"].value_counts(dropna=False))

print("\nDataset cleaning completed successfully.")