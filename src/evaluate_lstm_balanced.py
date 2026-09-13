import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


print("=" * 60)
print("ACDT PROJECT - EVALUATING BALANCED LSTM MODEL")
print("=" * 60)


# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

data = np.load("data/processed/lstm_train_test.npz")

X_test = data["X_test"]
y_test = data["y_test"]

print("\n[INFO] Test data loaded.")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# --------------------------------------------------
# 2. Load balanced LSTM model
# --------------------------------------------------

model_path = "models/learner_state_lstm_balanced.keras"

model = tf.keras.models.load_model(model_path)

print("\n[OK] Balanced LSTM model loaded.")


# --------------------------------------------------
# 3. Generate predictions
# --------------------------------------------------

print("\n[INFO] Generating predictions...")

probabilities = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

print("[OK] Predictions generated.")


# --------------------------------------------------
# 4. Class names
# --------------------------------------------------

class_names = [
    "stable_learning",
    "moderate_struggle",
    "high_struggle"
]


# --------------------------------------------------
# 5. Classification report
# --------------------------------------------------

print("\n" + "=" * 60)
print("BALANCED LSTM CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# --------------------------------------------------
# 6. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("BALANCED LSTM CONFUSION MATRIX")
print("=" * 60)

print(cm)


# --------------------------------------------------
# 7. Save report
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

report_path = "results/lstm_balanced_classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("BALANCED LSTM CLASSIFICATION REPORT\n")
    file.write("=" * 60 + "\n\n")
    file.write(report)
    file.write("\n\nCONFUSION MATRIX\n")
    file.write("=" * 60 + "\n")
    file.write(str(cm))

print("\n[OK] Classification report saved to:")
print(report_path)


# --------------------------------------------------
# 8. Save confusion matrix image
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8, 6))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=30,
    values_format="d"
)

plt.title("Balanced LSTM Learner-State Confusion Matrix")
plt.tight_layout()

figure_path = "results/lstm_balanced_confusion_matrix.png"

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n[OK] Confusion matrix image saved to:")
print(figure_path)

print("\n[OK] Balanced LSTM evaluation completed.")