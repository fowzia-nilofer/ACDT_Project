from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("ACDT PROJECT - MAJORITY CLASS BASELINE")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

test_path = (
    project_root
    / "data"
    / "processed"
    / "test.csv"
)

test_data = pd.read_csv(test_path)

y_test = test_data["correct"]

majority_class = y_test.mode()[0]

y_pred = [majority_class] * len(y_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nMost common class:", majority_class)
print(f"Majority baseline accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n[OK] Majority baseline evaluation completed.")