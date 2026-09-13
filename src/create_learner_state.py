from pathlib import Path
import pandas as pd
import numpy as np


print("=" * 60)
print("ACDT PROJECT - CREATING LEARNER STATES")
print("=" * 60)


project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "features.csv"
)

output_path = (
    project_root
    / "data"
    / "processed"
    / "learner_state.csv"
)


data = pd.read_csv(input_path)

print("\n[OK] Features dataset loaded.")
print(f"Dataset shape: {data.shape}")


# --------------------------------------------------
# 1. Handle missing values
# --------------------------------------------------

data["response_time_seconds"] = data[
    "response_time_seconds"
].fillna(
    data["response_time_seconds"].median()
)

data["hint_count"] = data["hint_count"].fillna(0)

data["attempt_count"] = data["attempt_count"].fillna(0)


# --------------------------------------------------
# 2. Normalize behavioral features
# --------------------------------------------------

def percentile_normalize(series):
    lower = series.quantile(0.01)
    upper = series.quantile(0.99)

    if upper == lower:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    clipped = series.clip(lower, upper)

    normalized = (
        (clipped - lower)
        / (upper - lower)
    )

    return normalized


data["response_time_normalized"] = (
    percentile_normalize(
        data["response_time_seconds"]
    )
)

data["hint_normalized"] = (
    percentile_normalize(
        data["hint_count"]
    )
)

data["attempt_normalized"] = (
    percentile_normalize(
        data["attempt_count"]
    )
)


# --------------------------------------------------
# 3. Create cognitive and behavioral scores
# --------------------------------------------------

data["knowledge_state"] = data["previous_accuracy"]


data["struggle_score"] = (
    0.40 * data["hint_normalized"]
    + 0.35 * data["attempt_normalized"]
    + 0.25 * (1 - data["knowledge_state"])
)


data["engagement_difficulty_score"] = (
    0.60 * data["response_time_normalized"]
    + 0.40 * data["hint_normalized"]
)


# --------------------------------------------------
# 4. Assign learner states
# --------------------------------------------------

data["learner_state"] = np.select(
    [
        data["previous_interactions"] == 0,
        data["struggle_score"] >= 0.65,
        data["struggle_score"] >= 0.35
    ],
    [
        "initial_state",
        "high_struggle",
        "moderate_struggle"
    ],
    default="stable_learning"
)


# --------------------------------------------------
# 5. Select output columns
# --------------------------------------------------

output_columns = [
    "user_id",
    "order_id",
    "problem_id",
    "correct",
    "previous_interactions",
    "previous_accuracy",
    "knowledge_state",
    "struggle_score",
    "engagement_difficulty_score",
    "response_time_normalized",
    "hint_normalized",
    "attempt_normalized",
    "learner_state"
]

output = data[output_columns]


# --------------------------------------------------
# 6. Save learner-state dataset
# --------------------------------------------------

output.to_csv(
    output_path,
    index=False
)


print("\n[OK] Learner-state dataset created.")
print(f"Saved to: {output_path}")
print(f"Output shape: {output.shape}")


print("\nLearner-state distribution:")
print(
    output["learner_state"]
    .value_counts()
)


print("\nAverage scores by learner state:")

summary = (
    output
    .groupby("learner_state")[
        [
            "knowledge_state",
            "struggle_score",
            "engagement_difficulty_score"
        ]
    ]
    .mean()
    .round(4)
)

print(summary)


print("\n[OK] Learner-state creation completed.")