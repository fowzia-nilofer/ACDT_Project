import os
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


print("=" * 60)
print("ACDT PROJECT - TRAINING BALANCED LSTM MODEL")
print("=" * 60)


# --------------------------------------------------
# 1. Load train/test data
# --------------------------------------------------

data = np.load("data/processed/lstm_train_test.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_test = data["X_test"]
y_test = data["y_test"]

print("\n[INFO] Data loaded.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# --------------------------------------------------
# 2. Calculate class weights
# --------------------------------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\n[INFO] Class weights:")

for class_id, weight in class_weights.items():
    print(f"Class {class_id}: {weight:.4f}")


# --------------------------------------------------
# 3. Build balanced LSTM model
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(10, 8)),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])


# --------------------------------------------------
# 4. Compile model
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n[INFO] Model architecture:")
model.summary()


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\n[INFO] Starting balanced LSTM training...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.10,
    epochs=15,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# 6. Evaluate on untouched test data
# --------------------------------------------------

print("\n[RESULT] Balanced LSTM evaluation...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=256,
    verbose=0
)

print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_accuracy, 4))
print(
    "Test accuracy percentage:",
    round(test_accuracy * 100, 2),
    "%"
)


# --------------------------------------------------
# 7. Save model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

model_path = "models/learner_state_lstm_balanced.keras"

model.save(model_path)

print("\n[OK] Balanced LSTM model saved to:")
print(model_path)

print("\n[OK] Balanced LSTM training completed.")