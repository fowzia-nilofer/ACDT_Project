from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 60)
print("ACDT PROJECT - ANALYZING SIMULATED FUTURES")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "models"
    / "transition_probability_model.csv"
)

output_path = (
    project_root
    / "results"
    / "simulated_future_analysis.csv"
)

data = pd.read_csv(input_path)

print("\n[OK] Transition probability model loaded.")

transition_model = {}

for current_state in data["learner_state"].unique():

    state_data = data[
        data["learner_state"] == current_state
    ]

    next_states = (
        state_data["next_learner_state"]
        .tolist()
    )

    probabilities = (
        state_data["transition_probability"]
        .to_numpy(dtype=float)
    )

    probabilities = (
        probabilities / probabilities.sum()
    )

    transition_model[current_state] = {
        "next_states": next_states,
        "probabilities": probabilities
    }


random_generator = np.random.default_rng(42)


def simulate_future(
    initial_state,
    number_of_steps=10
):

    current_state = initial_state
    future_states = []

    for step in range(number_of_steps):

        possible_next_states = (
            transition_model[current_state]["next_states"]
        )

        probabilities = (
            transition_model[current_state]["probabilities"]
        )

        next_state = random_generator.choice(
            possible_next_states,
            p=probabilities
        )

        future_states.append(next_state)

        current_state = next_state

    return future_states


initial_states = [
    "initial_state",
    "stable_learning",
    "moderate_struggle",
    "high_struggle"
]

number_of_simulations = 1000
number_of_steps = 10

results = []

for initial_state in initial_states:

    for simulation_number in range(
        number_of_simulations
    ):

        future = simulate_future(
            initial_state=initial_state,
            number_of_steps=number_of_steps
        )

        state_counts = pd.Series(
            future
        ).value_counts()

        results.append(
            {
                "initial_state": initial_state,
                "simulation_number": simulation_number,
                "stable_learning_count": state_counts.get(
                    "stable_learning",
                    0
                ),
                "moderate_struggle_count": state_counts.get(
                    "moderate_struggle",
                    0
                ),
                "high_struggle_count": state_counts.get(
                    "high_struggle",
                    0
                )
            }
        )

results_data = pd.DataFrame(results)

summary = (
    results_data
    .groupby("initial_state")
    [
        [
            "stable_learning_count",
            "moderate_struggle_count",
            "high_struggle_count"
        ]
    ]
    .mean()
    .reset_index()
)

summary["stable_learning_percentage"] = (
    summary["stable_learning_count"]
    / number_of_steps
    * 100
)

summary["moderate_struggle_percentage"] = (
    summary["moderate_struggle_count"]
    / number_of_steps
    * 100
)

summary["high_struggle_percentage"] = (
    summary["high_struggle_count"]
    / number_of_steps
    * 100
)

summary.to_csv(
    output_path,
    index=False
)

print("\n[OK] Simulated futures analyzed.")
print(f"Saved to: {output_path}")

print("\nAverage future-state distribution:")
print(
    summary[
        [
            "initial_state",
            "stable_learning_percentage",
            "moderate_struggle_percentage",
            "high_struggle_percentage"
        ]
    ].to_string(index=False)
)

print("\n[OK] Simulation analysis completed.")