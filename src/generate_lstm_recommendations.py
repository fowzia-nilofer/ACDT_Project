from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


print("=" * 60)
print("ACDT PROJECT - LSTM-BASED PERSONALIZED RECOMMENDATIONS")
print("=" * 60)


project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "processed" / "lstm_train_test.npz"
model_path = project_root / "models" / "learner_state_lstm_balanced.keras"
output_path = project_root / "results" / "lstm_personalized_recommendations.csv"


# Load test data
data = np.load(data_path)

X_test = data["X_test"]
y_test = data["y_test"]

print("\n[INFO] Test sequences loaded.")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# Load balanced LSTM model
model = load_model(model_path)

print("\n[OK] Balanced LSTM model loaded.")


# Generate predictions
print("\n[INFO] Generating learner-state predictions...")

predicted_probabilities = model.predict(X_test, verbose=0)
predicted_states = np.argmax(predicted_probabilities, axis=1)

print("[OK] Predictions generated.")


# State mapping
state_mapping = {
    0: "stable_learning",
    1: "moderate_struggle",
    2: "high_struggle"
}


# Recommendation rules
recommendation_rules = {
    "stable_learning": {
        "recommended_action": "normal_practice",
        "reason": "The learner is stable, so continue regular practice."
    },
    "moderate_struggle": {
        "recommended_action": "hint_support",
        "reason": "Provide hints to support the learner without immediately reducing difficulty."
    },
    "high_struggle": {
        "recommended_action": "concept_revision",
        "reason": "Review the underlying concept before continuing practice."
    }
}


# Create recommendation records
records = []

for index, predicted_state_id in enumerate(predicted_states):

    learner_state = state_mapping[int(predicted_state_id)]
    recommendation = recommendation_rules[learner_state]

    confidence = float(
        predicted_probabilities[index][predicted_state_id]
    )

    records.append(
        {
            "sequence_id": index,
            "predicted_learner_state": learner_state,
            "prediction_confidence": round(confidence, 4),
            "recommended_action": recommendation["recommended_action"],
            "reason": recommendation["reason"]
        }
    )


recommendations = pd.DataFrame(records)


# Save recommendations
recommendations.to_csv(
    output_path,
    index=False
)


print("\n[OK] LSTM-based recommendations created.")
print(f"Saved to: {output_path}")

print("\nPredicted state distribution:")
print(
    recommendations["predicted_learner_state"]
    .value_counts()
)

print("\nRecommended action distribution:")
print(
    recommendations["recommended_action"]
    .value_counts()
)

print("\nSample recommendations:")
print(
    recommendations.head(10).to_string(index=False)
)

print("\n[IMPORTANT]")
print(
    "These recommendations are prototype decision rules based "
    "on balanced LSTM predictions."
)

print("\n[OK] LSTM-based recommendation generation completed.")