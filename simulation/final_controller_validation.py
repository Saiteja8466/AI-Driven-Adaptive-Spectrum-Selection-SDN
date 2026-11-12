
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Reproducible experiment settings
N_CHANNELS = 8
N_SLOTS = 3000
TRAIN_SEEDS = range(30000, 30010)
TEST_SEEDS = range(40000, 40020)

SWITCHING_PENALTY = 1.0
INITIAL_CHANNEL = 1

FEATURES = [
    "pu_previous",
    "pu_current",
    "traffic_previous",
    "traffic_current",
    "interference_previous",
    "interference_current",
    "snr_db_current",
    "channel_id",
]


def generate_scenario(seed):
    """Generate one independent synthetic wireless scenario."""
    rng = np.random.default_rng(seed)
    all_rows = []

    for channel in range(1, N_CHANNELS + 1):
        pu = np.zeros(N_SLOTS)
        traffic = np.zeros(N_SLOTS)
        fading = np.zeros(N_SLOTS)
        pu_strength = np.zeros(N_SLOTS)
        snr = rng.uniform(-5, 20, N_SLOTS)

        pu[0] = rng.integers(0, 2)
        traffic[0] = rng.uniform(0, 1)
        fading[0] = rng.normal()
        pu_strength[0] = rng.uniform(4, 8)

        for t in range(1, N_SLOTS):
            # Primary-user activity: Markov process
            if pu[t - 1] == 1:
                pu[t] = 1 if rng.random() < 0.85 else 0
            else:
                pu[t] = 1 if rng.random() < 0.10 else 0

            # Correlated traffic load
            traffic[t] = (
                0.85 * traffic[t - 1]
                + 0.15 * rng.uniform(0, 1)
            )

            # Correlated fading
            fading[t] = (
                0.90 * fading[t - 1]
                + 0.10 * rng.normal()
            )

            # Slowly varying primary-user signal strength
            pu_strength[t] = (
                0.90 * pu_strength[t - 1]
                + 0.10 * rng.uniform(4, 8)
            )

        fading_gain = np.exp(0.25 * fading)

        interference = (
            1.0
            + pu * pu_strength * fading_gain
            + 0.5
            + 2.0 * traffic
        )

        for t in range(1, N_SLOTS - 1):
            all_rows.append({
                "time_slot": t,
                "channel_id": channel,
                "pu_previous": pu[t - 1],
                "pu_current": pu[t],
                "traffic_previous": traffic[t - 1],
                "traffic_current": traffic[t],
                "interference_previous": interference[t - 1],
                "interference_current": interference[t],
                "snr_db_current": snr[t],
                "future_interference": interference[t + 1],
            })

    return pd.DataFrame(all_rows)


def evaluate_controller(data, model=None):
    """Compare current-based and AI-based channel selection."""
    current_channel = INITIAL_CHANNEL
    ai_channel = INITIAL_CHANNEL

    current_costs = []
    ai_costs = []

    current_switches = 0
    ai_switches = 0

    for slot in sorted(data["time_slot"].unique()):
        frame = data[data["time_slot"] == slot].copy()
        frame = frame.sort_values("channel_id")

        # Current-based controller
        current_scores = (
            frame["interference_current"].to_numpy()
            + SWITCHING_PENALTY
            * (frame["channel_id"].to_numpy() != current_channel)
        )

        current_index = int(np.argmin(current_scores))
        selected_current = int(
            frame.iloc[current_index]["channel_id"]
        )

        actual_current = float(
            frame.iloc[current_index]["future_interference"]
        )

        if selected_current != current_channel:
            current_switches += 1

        current_channel = selected_current
        current_costs.append(actual_current)

        # AI-predicted controller
        predictions = model.predict(frame[FEATURES])

        ai_scores = (
            predictions
            + SWITCHING_PENALTY
            * (frame["channel_id"].to_numpy() != ai_channel)
        )

        ai_index = int(np.argmin(ai_scores))
        selected_ai = int(frame.iloc[ai_index]["channel_id"])

        actual_ai = float(
            frame.iloc[ai_index]["future_interference"]
        )

        if selected_ai != ai_channel:
            ai_switches += 1

        ai_channel = selected_ai
        ai_costs.append(actual_ai)

    total_slots = len(current_costs)

    return {
        "Current_Interference": np.mean(current_costs),
        "AI_Interference": np.mean(ai_costs),
        "Current_Switches": current_switches,
        "AI_Switches": ai_switches,
        "Current_Switching_Rate": (
            current_switches / total_slots * 100
        ),
        "AI_Switching_Rate": (
            ai_switches / total_slots * 100
        ),
    }


def main():
    os.makedirs("results", exist_ok=True)

    print("Generating independent training scenarios...")

    training_data = pd.concat(
        [generate_scenario(seed) for seed in TRAIN_SEEDS],
        ignore_index=True
    )

    # Train only on independent training scenarios
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        training_data[FEATURES],
        training_data["future_interference"]
    )

    print("Model training completed.")
    print("Starting validation on 20 unseen scenarios...")

    results = []

    for seed in TEST_SEEDS:
        test_data = generate_scenario(seed)

        # Evaluate on the later 30% of each test scenario
        test_data = test_data[
            test_data["time_slot"] >= int(N_SLOTS * 0.70)
        ]

        metrics = evaluate_controller(test_data, model)
        metrics["Test_Seed"] = seed
        results.append(metrics)

        print(
            f"Seed {seed}: "
            f"Current={metrics['Current_Interference']:.4f}, "
            f"AI={metrics['AI_Interference']:.4f}, "
            f"Current switches={metrics['Current_Switches']}, "
            f"AI switches={metrics['AI_Switches']}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/final_controller_validation.csv",
        index=False
    )

    summary = results_df.drop(columns="Test_Seed").agg(
        ["mean", "std"]
    ).T

    summary.to_csv(
        "results/final_controller_validation_summary.csv"
    )

    # Paired scenario-level difference and confidence interval
    differences = (
        results_df["AI_Interference"]
        - results_df["Current_Interference"]
    )

    mean_difference = differences.mean()
    standard_error = differences.std(ddof=1) / np.sqrt(len(differences))

    ci_low = mean_difference - 2.093 * standard_error
    ci_high = mean_difference + 2.093 * standard_error

    print("\n========== FINAL VALIDATION SUMMARY ==========")
    print(summary.round(4))

    print("\nAI minus Current interference:")
    print(f"Mean difference: {mean_difference:.4f}")
    print(f"Approx. 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    if ci_low <= 0 <= ci_high:
        print("Conclusion: No statistically clear difference.")
    elif mean_difference < 0:
        print("Conclusion: AI controller shows lower interference.")
    else:
        print("Conclusion: Current-based controller shows lower interference.")

    print("\nFiles saved:")
    print("results/final_controller_validation.csv")
    print("results/final_controller_validation_summary.csv")


if __name__ == "__main__":
    main()
