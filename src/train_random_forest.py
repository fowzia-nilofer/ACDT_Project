from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("=" * 60)
print("ACDT PROJECT - TRAINING RANDOM FOREST MODEL")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

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

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("\n[OK] Datasets loaded.")
print(f"Training shape: {train_data.shape}")
print(f"Testing shape : {test_data.shape}")

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

X_train = train_data[feature_columns]
y_train = train_data[target_column]

X_test = test_data[feature_columns]
y_test = test_data[target_column]

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("[OK] Model training completed.")

print("\nGenerating predictions...")

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature importance:")

for feature, importance in zip(
    feature_columns,
    model.feature_importances_
):
    print(f"{feature}: {importance:.4f}")

model_path = (
    project_root
    / "models"
    / "random_forest_model.pkl"
)

joblib.dump(model, model_path)

print(f"\n[OK] Model saved to: {model_path}")
print("\n[OK] Random Forest training completed successfully.")