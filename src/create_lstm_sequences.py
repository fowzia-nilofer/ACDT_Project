from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


print("=" * 60)
print("ACDT PROJECT - CREATING LSTM SEQUENCES")
print("=" * 60)


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

features_path = (
    project_root
    / "data"
    / "processed"
    / "features.csv"
)

learner_state_path = (
    project_root
    / "data"
    / "processed"
    / "learner_state.csv"
)

sequence_output_path = (
    project_root
    / "data"
    / "processed"
    / "lstm_sequences.npz"
)

scaler_output_path = (
    project_root
    / "models"
    / "lstm_feature_scaler.pkl"
)


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

features_data = pd.read_csv(features_path)

learner_state_data = pd.read_csv(
    learner_state_path
)

print("\n[INFO] Features dataset loaded.")
print("Features shape:", features_data.shape)

print("\n[INFO] Learner-state dataset loaded.")
print(
    "Learner-state shape:",
    learner_state_data.shape
)


# --------------------------------------------------
# 3. Select derived state features
# --------------------------------------------------

state_columns = [
    "knowledge_state",
    "struggle_score",
    "engagement_difficulty_score",
    "learner_state",
]

missing_state_columns = [
    column
    for column in state_columns
    if column not in learner_state_data.columns
]

if missing_state_columns:
    raise ValueError(
        "Missing learner-state columns: "
        + str(missing_state_columns)
    )


# --------------------------------------------------
# 4. Merge datasets
# --------------------------------------------------

merge_columns = [
    "user_id",
    "order_id",
]

data = features_data.merge(
    learner_state_data[
        merge_columns + state_columns
    ],
    on=merge_columns,
    how="inner",
    validate="one_to_one",
)

print("\n[INFO] Datasets merged.")
print("Merged shape:", data.shape)


# --------------------------------------------------
# 5. Sort chronologically
# --------------------------------------------------

data = data.sort_values(
    by=["user_id", "order_id"]
).reset_index(drop=True)


# --------------------------------------------------
# 6. Define LSTM input features
# --------------------------------------------------

feature_columns = [
    "correct",
    "response_time_seconds",
    "hint_count",
    "attempt_count",
    "previous_accuracy",
    "knowledge_state",
    "struggle_score",
    "engagement_difficulty_score",
]

target_column = "learner_state"


# --------------------------------------------------
# 7. Check required columns
# --------------------------------------------------

missing_feature_columns = [
    column
    for column in feature_columns
    if column not in data.columns
]

if missing_feature_columns:
    raise ValueError(
        "Missing feature columns: "
        + str(missing_feature_columns)
    )


# --------------------------------------------------
# 8. Handle missing values
# --------------------------------------------------

for column in feature_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    median_value = data[column].median()

    if pd.isna(median_value):
        median_value = 0

    data[column] = data[column].fillna(
        median_value
    )


# --------------------------------------------------
# 9. Keep valid prediction states
# --------------------------------------------------

valid_states = [
    "stable_learning",
    "moderate_struggle",
    "high_struggle",
]

data = data[
    data[target_column].isin(valid_states)
].copy()


# --------------------------------------------------
# 10. Encode target states
# --------------------------------------------------

state_to_number = {
    "stable_learning": 0,
    "moderate_struggle": 1,
    "high_struggle": 2,
}

data["target_encoded"] = data[
    target_column
].map(state_to_number)


# --------------------------------------------------
# 11. Scale input features
# --------------------------------------------------

scaler = StandardScaler()

data[feature_columns] = scaler.fit_transform(
    data[feature_columns]
)

joblib.dump(
    scaler,
    scaler_output_path
)

print("\n[OK] Feature scaler saved to:")
print(scaler_output_path)


# --------------------------------------------------
# 12. Create fixed-length sequences
# --------------------------------------------------

sequence_length = 10

X_sequences = []
y_targets = []
sequence_user_ids = []

learner_count = 0


for user_id, learner_data in data.groupby("user_id"):

    learner_data = learner_data.sort_values(
        "order_id"
    )

    feature_values = learner_data[
        feature_columns
    ].to_numpy(dtype=np.float32)

    target_values = learner_data[
        "target_encoded"
    ].to_numpy(dtype=np.int64)

    if len(learner_data) <= sequence_length:
        continue

    learner_count += 1

    for index in range(
        sequence_length,
        len(learner_data)
    ):

        sequence = feature_values[
            index - sequence_length:index
        ]

        target = target_values[index]

        X_sequences.append(sequence)
        y_targets.append(target)
        sequence_user_ids.append(user_id)


# --------------------------------------------------
# 13. Convert to NumPy arrays
# --------------------------------------------------

X_sequences = np.asarray(
    X_sequences,
    dtype=np.float32
)

y_targets = np.asarray(
    y_targets,
    dtype=np.int64
)

sequence_user_ids = np.asarray(
    sequence_user_ids
)


# --------------------------------------------------
# 14. Validate generated arrays
# --------------------------------------------------

if not (
    len(X_sequences)
    == len(y_targets)
    == len(sequence_user_ids)
):
    raise ValueError(
        "X, y, and user_ids lengths do not match."
    )

if len(X_sequences) == 0:
    raise ValueError(
        "No LSTM sequences were created."
    )


# --------------------------------------------------
# 15. Save sequences
# --------------------------------------------------

np.savez_compressed(
    sequence_output_path,
    X=X_sequences,
    y=y_targets,
    user_ids=sequence_user_ids
)


# --------------------------------------------------
# 16. Display summary
# --------------------------------------------------

print("\n[OK] LSTM sequences created.")

print("Learners used:", learner_count)

print(
    "Unique sequence learners:",
    len(np.unique(sequence_user_ids))
)

print("Input shape:", X_sequences.shape)

print("Target shape:", y_targets.shape)

print(
    "User ID shape:",
    sequence_user_ids.shape
)

print("\nTarget distribution:")

target_distribution = (
    pd.Series(y_targets)
    .value_counts()
    .sort_index()
)

for state_number, count in target_distribution.items():

    state_name = valid_states[state_number]

    print(
        state_name,
        ":",
        count
    )

print("\nSaved to:")
print(sequence_output_path)

print("\n[OK] Sequence-generation completed.")