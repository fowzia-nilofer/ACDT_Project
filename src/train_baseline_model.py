from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("=" * 60)
print("ACDT PROJECT - TRAINING BASELINE MODEL")
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

print("\nLoading training data...")
train_data = pd.read_csv(train_path)

print("Loading testing data...")
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

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("[OK] Model training completed.")

print("\nGenerating predictions...")

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature coefficients:")

for feature, coefficient in zip(
    feature_columns,
    model.coef_[0]
):
    print(f"{feature}: {coefficient:.4f}")

model_path = (
    project_root
    / "models"
    / "baseline_logistic_regression.pkl"
)

import joblib

joblib.dump(model, model_path)

print(f"\n[OK] Model saved to: {model_path}")
print("\n[OK] Baseline model training completed successfully.")