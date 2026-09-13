from pathlib import Path
import pandas as pd

print("=" * 60)
print("ACDT PROJECT - CREATING PERSONALIZED RECOMMENDATIONS")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

output_path = (
    project_root
    / "results"
    / "personalized_recommendations.csv"
)

recommendation_rules = {
    "initial_state": {
        "recommended_action": "normal_practice",
        "reason": "Start with normal practice to establish a learning baseline."
    },
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

records = []

for learner_state, recommendation in recommendation_rules.items():

    records.append(
        {
            "learner_state": learner_state,
            "recommended_action": recommendation[
                "recommended_action"
            ],
            "reason": recommendation["reason"]
        }
    )

recommendations = pd.DataFrame(records)

recommendations.to_csv(
    output_path,
    index=False
)

print("\n[OK] Personalized recommendations created.")
print(f"Saved to: {output_path}")

print("\nRecommendations:")
print(
    recommendations.to_string(index=False)
)

print("\n[IMPORTANT]")
print(
    "These recommendations are prototype decision rules."
)
print(
    "They are not experimentally validated interventions."
)

print("\n[OK] Personalized recommendation creation completed.")