import os
import matplotlib.pyplot as plt
import numpy as np


# Model results
models = [
    "Original LSTM",
    "Balanced LSTM"
]

accuracy = [87.99, 71.03]
macro_f1 = [45.20, 44.45]

moderate_recall = [30.10, 58.80]
high_recall = [0.00, 35.46]


# Create results directory
os.makedirs("results", exist_ok=True)


# Comparison 1: Accuracy and Macro F1
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width / 2, accuracy, width, label="Accuracy (%)")
plt.bar(x + width / 2, macro_f1, width, label="Macro F1 (%)")

plt.xticks(x, models)
plt.ylabel("Score (%)")
plt.title("Original LSTM vs Balanced LSTM")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/lstm_model_comparison.png",
    dpi=300
)

plt.show()


# Comparison 2: Recall for struggling learners
plt.figure(figsize=(8, 5))

plt.bar(x - width / 2, moderate_recall, width, label="Moderate Struggle Recall (%)")
plt.bar(x + width / 2, high_recall, width, label="High Struggle Recall (%)")

plt.xticks(x, models)
plt.ylabel("Recall (%)")
plt.title("Struggle Detection Comparison")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/lstm_struggle_recall_comparison.png",
    dpi=300
)

plt.show()


print("=" * 60)
print("LSTM MODEL COMPARISON COMPLETED")
print("=" * 60)

print("\nComparison charts saved:")
print("results/lstm_model_comparison.png")
print("results/lstm_struggle_recall_comparison.png")