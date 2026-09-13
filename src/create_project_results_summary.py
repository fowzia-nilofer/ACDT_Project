from pathlib import Path
import pandas as pd


print("=" * 60)
print("ACDT PROJECT - CREATING PROJECT RESULTS SUMMARY")
print("=" * 60)


project_root = Path(__file__).resolve().parent.parent

output_path = (
    project_root
    / "results"
    / "project_results_summary.csv"
)


results = [
    {
        "category": "Dataset",
        "metric": "Total interactions",
        "value": 346860,
        "description": "Number of learner interaction records"
    },
    {
        "category": "Dataset",
        "metric": "Total learners",
        "value": 4217,
        "description": "Number of unique learners"
    },
    {
        "category": "Dataset",
        "metric": "Total skills",
        "value": 149,
        "description": "Number of distinct skills including unknown skill"
    },
    {
        "category": "Prediction",
        "metric": "Logistic Regression accuracy",
        "value": 91.32,
        "description": "Baseline prediction accuracy in percentage"
    },
    {
        "category": "Prediction",
        "metric": "Random Forest accuracy",
        "value": 93.39,
        "description": "Random Forest prediction accuracy in percentage"
    },
    {
        "category": "Prediction",
        "metric": "Random Forest ROC-AUC",
        "value": 0.9462,
        "description": "Random Forest ROC-AUC score"
    },
    {
        "category": "Learner States",
        "metric": "Stable learning learners",
        "value": 3359,
        "description": "Learners receiving normal practice"
    },
    {
        "category": "Learner States",
        "metric": "Moderate struggle learners",
        "value": 659,
        "description": "Learners receiving hint support"
    },
    {
        "category": "Learner States",
        "metric": "Initial state learners",
        "value": 120,
        "description": "Learners with limited interaction history"
    },
    {
        "category": "Learner States",
        "metric": "High struggle learners",
        "value": 79,
        "description": "Learners receiving concept revision"
    },
    {
        "category": "Simulation",
        "metric": "Simulation steps",
        "value": 10,
        "description": "Number of future steps simulated"
    },
    {
        "category": "Simulation",
        "metric": "Simulations per initial state",
        "value": 1000,
        "description": "Monte Carlo simulations for each initial state"
    },
    {
        "category": "Recommendation",
        "metric": "Learners with recommendations",
        "value": 4217,
        "description": "One recommendation generated for each learner"
    }
]


summary = pd.DataFrame(results)

summary.to_csv(
    output_path,
    index=False
)

print("\n[OK] Project results summary created.")
print(f"Saved to: {output_path}")

print("\nProject Results:")
print(summary.to_string(index=False))

print("\n[OK] Project results summary completed.")