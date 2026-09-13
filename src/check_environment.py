from pathlib import Path
import sys


print("=" * 60)
print("ACDT PROJECT - ENVIRONMENT CHECK")
print("=" * 60)

# Find the main project folder
project_root = Path(__file__).resolve().parent.parent

print(f"\nProject folder:")
print(project_root)

print(f"\nPython version:")
print(sys.version)

print("\nChecking project folders...")

required_folders = [
    project_root / "data",
    project_root / "data" / "raw",
    project_root / "data" / "processed",
    project_root / "src",
    project_root / "models",
    project_root / "results",
    project_root / "notebooks",
]

for folder in required_folders:
    if folder.exists():
        print(f"[OK] {folder.relative_to(project_root)}")
    else:
        print(f"[MISSING] {folder.relative_to(project_root)}")

print("\nChecking installed libraries...")

libraries = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
]

for library in libraries:
    try:
        __import__(library)
        print(f"[OK] {library}")
    except ImportError:
        print(f"[MISSING] {library}")

print("\nEnvironment check completed.")