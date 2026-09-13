from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


print("=" * 60)
print("ACDT PROJECT - TRAINING LSTM MODEL")
print("=" * 60)


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root
    / "data"
    / "processed"
    / "lstm_train_test.npz"
)

model_path = (
    project_root
    / "models"
    / "learner_state_lstm.keras"
)


# --------------------------------------------------
# 2. Set reproducibility
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)


# --------------------------------------------------
# 3. Load train/test data
# --------------------------------------------------

data = np.load(data_path)

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
# 4. Convert labels to one-hot encoding
# --------------------------------------------------

number_of_classes = 3

y_train_encoded = to_categorical(
    y_train,
    num_classes=number_of_classes
)

y_test_encoded = to_categorical(
    y_test,
    num_classes=number_of_classes
)


# --------------------------------------------------
# 5. Build LSTM model
# --------------------------------------------------

model = Sequential(
    [
        LSTM(
            64,
            input_shape=(
                X_train.shape[1],
                X_train.shape[2]
            ),
            return_sequences=False
        ),

        Dropout(0.30),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.20),

        Dense(
            number_of_classes,
            activation="softmax"
        ),
    ]
)


# --------------------------------------------------
# 6. Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n[INFO] Model architecture:")

model.summary()


# --------------------------------------------------
# 7. Define early stopping
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# --------------------------------------------------
# 8. Train model
# --------------------------------------------------

print("\n[INFO] Starting LSTM training...")

history = model.fit(
    X_train,
    y_train_encoded,

    validation_data=(
        X_test,
        y_test_encoded
    ),

    epochs=15,
    batch_size=256,
    callbacks=[
        early_stopping
    ],

    verbose=1
)


# --------------------------------------------------
# 9. Evaluate model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_encoded,
    verbose=0
)

print("\n[RESULT] LSTM evaluation completed.")

print(
    "Test loss:",
    round(test_loss, 4)
)

print(
    "Test accuracy:",
    round(test_accuracy, 4)
)

print(
    "Test accuracy percentage:",
    round(test_accuracy * 100, 2),
    "%"
)


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

model.save(model_path)

print("\n[OK] LSTM model saved to:")
print(model_path)

print("\n[OK] LSTM training completed.")