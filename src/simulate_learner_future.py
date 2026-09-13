from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 60)
print("ACDT PROJECT - SIMULATING LEARNER FUTURE")
print("=" * 60)

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "models"
    / "transition_probability_model.csv"
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

print("\n[OK] Transition model prepared.")

random_generator = np.random.default_rng(42)


def simulate_future(
    initial_state,
    number_of_steps=10
):

    current_state = initial_state
    future_states = [current_state]

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

for initial_state in initial_states:

    future = simulate_future(
        initial_state=initial_state,
        number_of_steps=10
    )

    print(f"\nStarting state: {initial_state}")
    print("Simulated future:")
    print(" -> ".join(future))

print("\n[OK] Future simulation completed.")