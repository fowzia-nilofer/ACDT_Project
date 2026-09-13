from pathlib import Path
import numpy as np
import pandas as pd


print("=" * 60)
print("ACDT PROJECT - EVALUATING ACTION SIMULATIONS")
print("=" * 60)


project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root
    / "models"
    / "action_transition_model.csv"
)

output_path = (
    project_root
    / "results"
    / "action_simulation_comparison.csv"
)


data = pd.read_csv(model_path)

states = sorted(
    data["learner_state"].unique()
)

actions = sorted(
    data["action"].unique()
)

next_states = sorted(
    data["next_learner_state"].unique()
)

state_to_index = {
    state: index
    for index, state in enumerate(next_states)
}


def get_transition_matrix(current_state, action):
    selected = data[
        (data["learner_state"] == current_state)
        & (data["action"] == action)
    ]

    matrix = np.zeros(
        len(next_states),
        dtype=float
    )

    for _, row in selected.iterrows():
        next_state = row["next_learner_state"]

        matrix[state_to_index[next_state]] = (
            row["adjusted_probability"]
        )

    matrix = matrix / matrix.sum()

    return matrix


results = []

number_of_simulations = 1000
number_of_steps = 10

random_generator = np.random.default_rng(42)


for initial_state in states:

    for action in actions:

        high_struggle_percentages = []

        for simulation in range(
            number_of_simulations
        ):

            current_state = initial_state
            high_struggle_count = 0

            for step in range(number_of_steps):

                probabilities = get_transition_matrix(
                    current_state,
                    action
                )

                next_state = random_generator.choice(
                    next_states,
                    p=probabilities
                )

                if next_state == "high_struggle":
                    high_struggle_count += 1

                current_state = next_state

            high_struggle_percentage = (
                high_struggle_count
                / number_of_steps
                * 100
            )

            high_struggle_percentages.append(
                high_struggle_percentage
            )

        average_high_struggle = np.mean(
            high_struggle_percentages
        )

        results.append(
            {
                "initial_state": initial_state,
                "action": action,
                "average_high_struggle_percentage": round(
                    average_high_struggle,
                    2
                )
            }
        )


results_data = pd.DataFrame(results)

results_data = results_data.sort_values(
    [
        "initial_state",
        "average_high_struggle_percentage"
    ]
)

results_data.to_csv(
    output_path,
    index=False
)

print("\nAction simulation comparison:")
print(
    results_data.to_string(index=False)
)

print("\n[OK] Action simulation comparison created.")
print(f"Saved to: {output_path}")

print("\n[OK] Action simulation evaluation completed.")