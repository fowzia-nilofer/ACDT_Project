from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - CREATING ACTION-CONDITIONED MODEL")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

transition_path = (
    project_root
    / "models"
    / "transition_probability_model.csv"
)

actions_path = (
    project_root
    / "models"
    / "learning_actions.csv"
)

output_path = (
    project_root
    / "models"
    / "action_transition_model.csv"
)

transitions = pd.read_csv(transition_path)
actions = pd.read_csv(actions_path)

print("\n[OK] Transition model loaded.")
print("[OK] Learning actions loaded.")

records = []

for _, transition in transitions.iterrows():

    current_state = transition["learner_state"]
    next_state = transition["next_learner_state"]
    base_probability = transition["transition_probability"]

    for _, action in actions.iterrows():

        action_name = action["action"]
        reduction = action["hypothetical_struggle_reduction"]

        adjusted_probability = base_probability

        if (
            action_name != "normal_practice"
            and next_state == "high_struggle"
        ):
            adjusted_probability = (
                base_probability * (1 - reduction)
            )

        records.append(
            {
                "learner_state": current_state,
                "action": action_name,
                "next_learner_state": next_state,
                "base_probability": base_probability,
                "hypothetical_reduction": reduction,
                "adjusted_probability": adjusted_probability
            }
        )

action_model = pd.DataFrame(records)

action_model["adjusted_probability"] = (
    action_model
    .groupby(
        ["learner_state", "action"]
    )["adjusted_probability"]
    .transform(
        lambda probabilities:
        probabilities / probabilities.sum()
    )
)

action_model["adjusted_probability"] = (
    action_model["adjusted_probability"].round(4)
)

action_model.to_csv(
    output_path,
    index=False
)

print("\n[OK] Action-conditioned transition model created.")
print(f"Saved to: {output_path}")

print("\nModel shape:")
print(action_model.shape)

print("\nExample action-conditioned transitions:")
print(
    action_model[
        [
            "learner_state",
            "action",
            "next_learner_state",
            "adjusted_probability"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n[IMPORTANT]")
print(
    "Adjusted probabilities are hypothetical prototype values."
)
print(
    "They are not experimentally validated intervention effects."
)

print("\n[OK] Action-conditioned model creation completed.")