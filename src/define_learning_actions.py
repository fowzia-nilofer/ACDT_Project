from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - DEFINING LEARNING ACTIONS")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

output_path = (
    project_root
    / "models"
    / "learning_actions.csv"
)

actions = pd.DataFrame(
    [
        {
            "action": "normal_practice",
            "description": "Continue with normal practice questions",
            "target_state": "all_states",
            "hypothetical_struggle_reduction": 0.00
        },
        {
            "action": "easy_practice",
            "description": "Provide easier practice questions",
            "target_state": "moderate_or_high_struggle",
            "hypothetical_struggle_reduction": 0.15
        },
        {
            "action": "hint_support",
            "description": "Provide additional hints or guidance",
            "target_state": "moderate_or_high_struggle",
            "hypothetical_struggle_reduction": 0.20
        },
        {
            "action": "concept_revision",
            "description": "Recommend revision of the current concept",
            "target_state": "high_struggle",
            "hypothetical_struggle_reduction": 0.25
        }
    ]
)

actions.to_csv(
    output_path,
    index=False
)

print("\n[OK] Learning actions defined.")
print(f"Saved to: {output_path}")

print("\nAvailable actions:")
print(actions.to_string(index=False))

print("\n[IMPORTANT]")
print(
    "The intervention effects are hypothetical prototype values."
)
print(
    "They must not be presented as experimentally validated results."
)

print("\n[OK] Learning-action definition completed.")