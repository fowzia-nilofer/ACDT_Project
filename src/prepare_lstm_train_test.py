from pathlib import Path

import numpy as np


print("=" * 60)
print("ACDT PROJECT - PREPARING LSTM TRAIN/TEST DATA")
print("=" * 60)


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

sequence_path = (
    project_root
    / "data"
    / "processed"
    / "lstm_sequences.npz"
)

output_path = (
    project_root
    / "data"
    / "processed"
    / "lstm_train_test.npz"
)


# --------------------------------------------------
# 2. Load sequences
# --------------------------------------------------

data = np.load(sequence_path)

X = data["X"]
y = data["y"]
user_ids = data["user_ids"]

print("\n[INFO] LSTM data loaded.")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("User ID shape:", user_ids.shape)


# --------------------------------------------------
# 3. Get unique learners
# --------------------------------------------------

unique_learners = np.unique(user_ids)

print(
    "Unique learners:",
    len(unique_learners)
)


# --------------------------------------------------
# 4. Shuffle learners reproducibly
# --------------------------------------------------

random_generator = np.random.default_rng(
    seed=42
)

shuffled_learners = unique_learners.copy()

random_generator.shuffle(
    shuffled_learners
)


# --------------------------------------------------
# 5. Split learners
# --------------------------------------------------

train_ratio = 0.80

split_index = int(
    len(shuffled_learners) * train_ratio
)

train_learners = shuffled_learners[
    :split_index
]

test_learners = shuffled_learners[
    split_index:
]


# --------------------------------------------------
# 6. Create learner masks
# --------------------------------------------------

train_mask = np.isin(
    user_ids,
    train_learners
)

test_mask = np.isin(
    user_ids,
    test_learners
)


# --------------------------------------------------
# 7. Create train/test arrays
# --------------------------------------------------

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

user_ids_train = user_ids[train_mask]
user_ids_test = user_ids[test_mask]


# --------------------------------------------------
# 8. Validate split
# --------------------------------------------------

train_unique = set(
    np.unique(user_ids_train)
)

test_unique = set(
    np.unique(user_ids_test)
)

overlap = train_unique.intersection(
    test_unique
)

if overlap:
    raise ValueError(
        "Learner leakage detected: "
        + str(overlap)
    )


# --------------------------------------------------
# 9. Save split data
# --------------------------------------------------

np.savez_compressed(
    output_path,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    user_ids_train=user_ids_train,
    user_ids_test=user_ids_test,
)


# --------------------------------------------------
# 10. Display summary
# --------------------------------------------------

print("\n[OK] Learner-level split created.")

print(
    "Training learners:",
    len(train_unique)
)

print(
    "Testing learners:",
    len(test_unique)
)

print(
    "Training sequences:",
    len(X_train)
)

print(
    "Testing sequences:",
    len(X_test)
)

print(
    "Learner overlap:",
    len(overlap)
)

print("\nSaved to:")
print(output_path)

print("\n[OK] LSTM train/test preparation completed.")