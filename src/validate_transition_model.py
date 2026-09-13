from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - VALIDATING TRANSITION MODEL")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "models"
    / "transition_probability_model.csv"
)

data = pd.read_csv(input_path)

print("\n[OK] Transition probability model loaded.")
print(f"Dataset shape: {data.shape}")

probability_sums = (
    data
    .groupby("learner_state")["transition_probability"]
    .sum()
    .reset_index()
)

probability_sums = probability_sums.rename(
    columns={
        "transition_probability": "probability_sum"
    }
)

print("\nProbability sum for each current learner state:")
print(probability_sums.to_string(index=False))

print("\nExpected result:")
print("Each probability sum should be approximately 1.0.")

invalid_states = probability_sums[
    (probability_sums["probability_sum"] < 0.999)
    |
    (probability_sums["probability_sum"] > 1.001)
]

if len(invalid_states) == 0:
    print("\n[OK] All transition probabilities are valid.")
else:
    print("\n[WARNING] Some probability sums are outside the expected range:")
    print(invalid_states.to_string(index=False))

print("\n[OK] Transition-model validation completed.")