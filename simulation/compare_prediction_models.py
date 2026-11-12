
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from final_controller_validation import (
    generate_scenario,
    FEATURES,
    TRAIN_SEEDS,
    TEST_SEEDS,
    SWITCHING_PENALTY,
    INITIAL_CHANNEL,
)


def evaluate_all_controllers(data, models):
    channels = {
        "Current-based": INITIAL_CHANNEL,
        "Random Forest": INITIAL_CHANNEL,
        "Linear Regression": INITIAL_CHANNEL,
    }

    costs = {name: [] for name in channels}
    switches = {name: 0 for name in channels}

    for slot in sorted(data["time_slot"].unique()):
        frame = data[data["time_slot"] == slot].copy()
        frame = frame.sort_values("channel_id")

        channel_ids = frame["channel_id"].to_numpy()
        current_values = frame["interference_current"].to_numpy()
        actual_values = frame["future_interference"].to_numpy()

        # Current-based baseline
        scores = current_values + SWITCHING_PENALTY * (
            channel_ids != channels["Current-based"]
        )
        index = int(np.argmin(scores))
        selected = int(channel_ids[index])

        if selected != channels["Current-based"]:
            switches["Current-based"] += 1

        channels["Current-based"] = selected
        costs["Current-based"].append(actual_values[index])

        # AI models
        for name, model in models.items():
            predictions = model.predict(frame[FEATURES])

            scores = predictions + SWITCHING_PENALTY * (
                channel_ids != channels[name]
            )

            index = int(np.argmin(scores))
            selected = int(channel_ids[index])

            if selected != channels[name]:
                switches[name] += 1

            channels[name] = selected
            costs[name].append(actual_values[index])

    total_slots = len(costs["Current-based"])

    results = []

    for name in costs:
        results.append({
            "Controller": name,
            "Average_Interference": np.mean(costs[name]),
            "Channel_Switches": switches[name],
            "Switching_Rate_Percent": (
                switches[name] / total_slots * 100
            ),
        })

    return results


def main():
    os.makedirs("results", exist_ok=True)

    print("Generating training scenarios...")

    training_data = pd.concat(
        [generate_scenario(seed) for seed in TRAIN_SEEDS],
        ignore_index=True
    )

    X_train = training_data[FEATURES]
    y_train = training_data["future_interference"]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),
        "Linear Regression": LinearRegression(),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    all_results = []

    print("\nTesting on the same 20 scenarios...")

    for seed in TEST_SEEDS:
        test_data = generate_scenario(seed)

        test_data = test_data[
            test_data["time_slot"] >= 2100
        ]

        scenario_results = evaluate_all_controllers(
            test_data, models
        )

        for result in scenario_results:
            result["Test_Seed"] = seed
            all_results.append(result)

        print(f"Completed test scenario: {seed}")

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        "results/model_comparison_validation.csv",
        index=False
    )

    summary = (
        results_df.groupby("Controller")
        .agg({
            "Average_Interference": ["mean", "std"],
            "Channel_Switches": ["mean", "std"],
            "Switching_Rate_Percent": ["mean", "std"],
        })
        .round(4)
    )

    summary.to_csv(
        "results/model_comparison_summary.csv"
    )

    print("\n========== MODEL COMPARISON ==========")
    print(summary)

    # Paired comparison against current-based controller
    pivot = results_df.pivot(
        index="Test_Seed",
        columns="Controller",
        values="Average_Interference"
    )

    print("\n========== PAIRED DIFFERENCES ==========")

    for name in ["Random Forest", "Linear Regression"]:
        differences = (
            pivot[name] - pivot["Current-based"]
        )

        mean_diff = differences.mean()
        se = differences.std(ddof=1) / np.sqrt(len(differences))

        ci_low = mean_diff - 2.093 * se
        ci_high = mean_diff + 2.093 * se

        print(f"\n{name} minus Current-based:")
        print(f"Mean difference: {mean_diff:.4f}")
        print(f"Approx. 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        if ci_low <= 0 <= ci_high:
            print("No statistically clear difference.")
        elif mean_diff < 0:
            print("Lower interference than current-based.")
        else:
            print("Higher interference than current-based.")

    print("\nSaved:")
    print("results/model_comparison_validation.csv")
    print("results/model_comparison_summary.csv")


if __name__ == "__main__":
    main()
