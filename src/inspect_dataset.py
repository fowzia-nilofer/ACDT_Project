from pathlib import Path
import pandas as pd


print("=" * 60)
print("ACDT PROJECT - DATASET INSPECTION")
print("=" * 60)

# Locate the project folder
project_root = Path(__file__).resolve().parent.parent

# Expected dataset location
dataset_path = (
    project_root
    / "data"
    / "raw"
    / "assistments_skill_builder.csv"
)

print("\nExpected dataset path:")
print(dataset_path)

# Check whether the file exists
if not dataset_path.exists():
    print("\nERROR: Dataset file was not found.")
    print("Please check the filename and location.")
    raise SystemExit

print("\n[OK] Dataset file found.")

# Display file size
file_size_mb = dataset_path.stat().st_size / (1024 * 1024)

print(f"File size: {file_size_mb:.2f} MB")

print("\nReading dataset header...")

# Read only the first five rows initially
sample_data = pd.read_csv(
    dataset_path,
    nrows=5,
    low_memory=False
)

print("\nColumn names:")
for index, column in enumerate(sample_data.columns, start=1):
    print(f"{index}. {column}")

print("\nFirst five rows:")
print(sample_data.to_string(index=False))

print("\nChecking total number of rows...")

# Count rows without loading the complete dataset
with open(dataset_path, "r", encoding="utf-8", errors="ignore") as file:
    total_lines = sum(1 for _ in file)

total_rows = total_lines - 1

print(f"Total data rows: {total_rows:,}")

print("\nDataset inspection completed.")