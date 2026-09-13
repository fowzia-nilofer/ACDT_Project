from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - PREPARING MODELING DATASET")
print("=" * 60)

# --------------------------------------------------
# 1. Locate cleaned dataset
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
print(f"Original shape: {data.shape}")

# --------------------------------------------------
# 3. Handle missing skill values
# --------------------------------------------------

missing_skill_ids = data["skill_id"].isna().sum()
missing_skill_names = data["skill_name"].isna().sum()

print("\nMissing skill IDs before handling:", missing_skill_ids)
print("Missing skill names before handling:", missing_skill_names)

data["skill_id"] = data["skill_id"].fillna("unknown_skill")
data["skill_name"] = data["skill_name"].fillna("Unknown Skill")

# --------------------------------------------------
# 4. Create useful modeling columns
# --------------------------------------------------

# Ensure correct is numeric
data["correct"] = pd.to_numeric(
    data["correct"],
    errors="coerce"
)

# Convert response time from milliseconds to seconds
data["response_time_seconds"] = (
    data["ms_first_response"] / 1000
)

# Replace invalid or negative response times
data.loc[
    data["response_time_seconds"] < 0,
    "response_time_seconds"
] = pd.NA

# --------------------------------------------------
# 5. Sort interactions chronologically per student
# --------------------------------------------------

data = data.sort_values(
    by=["user_id", "order_id"]
).reset_index(drop=True)

# --------------------------------------------------
# 6. Save modeling dataset
# --------------------------------------------------

output_path = (
    project_root
    / "data"
    / "processed"
    / "modeling_interactions.csv"
)

data.to_csv(output_path, index=False)

# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODELING DATASET PREPARED")
print("=" * 60)

print(f"Final shape: {data.shape}")
print(f"Saved file: {output_path}")

print("\nRemaining missing values:")
print(
    data[
        ["skill_id", "skill_name", "correct",
         "response_time_seconds"]
    ].isnull().sum()
)

print("\nNew column added:")
print("- response_time_seconds")

print("\n[OK] Modeling dataset preparation completed.")