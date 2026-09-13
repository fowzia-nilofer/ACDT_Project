from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


print("=" * 60)
print("ACDT PROJECT - VISUALIZING ACTION SIMULATIONS")
print("=" * 60)


project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "results"
    / "action_simulation_comparison.csv"
)

output_path = (
    project_root
    / "results"
    / "action_simulation_comparison.png"
)


data = pd.read_csv(input_path)

states = data["initial_state"].unique()


for state in states:

    state_data = data[
        data["initial_state"] == state
    ].sort_values(
        "average_high_struggle_percentage"
    )

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        state_data["action"],
        state_data[
            "average_high_struggle_percentage"
        ]
    )

    plt.title(
        "Action Comparison from "
        + state
    )

    plt.xlabel("Learning Action")

    plt.ylabel(
        "Average Future High-Struggle Percentage"
    )

    plt.xticks(rotation=20)

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    safe_state_name = state.replace(
        "_",
        "-"
    )

    state_output_path = (
        project_root
        / "results"
        / (
            "action_comparison_"
            + safe_state_name
            + ".png"
        )
    )

    plt.savefig(
        state_output_path,
        dpi=300
    )

    plt.show()

    print(
        "\n[OK] Chart created for:",
        state
    )

    print(
        "Saved to:",
        state_output_path
    )


print(
    "\n[OK] Action simulation visualizations completed."
)