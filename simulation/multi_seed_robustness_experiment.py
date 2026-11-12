
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from scipy.stats import t

# ============================================================
# 20-SEED ROBUSTNESS VALIDATION
# Research: AI-Driven Adaptive Spectrum Selection
# ============================================================

NUM_SEEDS = 20
NUM_CHANNELS = 8
TIME_SLOTS = 3000

SWITCHING_PENALTY = 0.5
FEATURES = [
    "channel_id",
    "pu_occupancy_previous",
    "pu_occupancy_current",
    "traffic_load_previous",
    "traffic_load_current",
    "interference_previous",
    "interference_current",
    "snr_db_current"
]

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# ------------------------------------------------------------
# Generate one independent synthetic scenario
# ------------------------------------------------------------

def generate_scenario(seed):

    rng = np.random.default_rng(seed)
    all_data = []

    for channel in range(NUM_CHANNELS):

        # Primary User activity: Markov model
        occupancy = np.zeros(TIME_SLOTS, dtype=int)
        occupancy[0] = rng.integers(0, 2)

        for step in range(1, TIME_SLOTS):
            if occupancy[step - 1] == 1:
                occupancy[step] = (
                    0 if rng.random() < 0.15 else 1
                )
            else:
                occupancy[step] = (
                    1 if rng.random() < 0.10 else 0
                )

        # Temporally correlated traffic
        traffic = np.zeros(TIME_SLOTS)
        traffic[0] = rng.uniform(0.1, 0.9)

        for step in range(1, TIME_SLOTS):
            traffic[step] = (
                0.85 * traffic[step - 1]
                + 0.15 * rng.uniform(0, 1)
            )

        # Correlated Rayleigh-style fading process
        fading_state = np.zeros(TIME_SLOTS)
        fading_state[0] = rng.normal()

        correlation = 0.90

        for step in range(1, TIME_SLOTS):
            fading_state[step] = (
                correlation * fading_state[step - 1]
                + np.sqrt(1 - correlation ** 2)
                * rng.normal()
            )

        fading = np.exp(0.25 * fading_state)

        # Slowly varying Primary User interference strength
        pu_strength = np.zeros(TIME_SLOTS)
        pu_strength[0] = rng.uniform(4, 8)

        for step in range(1, TIME_SLOTS):
            pu_strength[step] = (
                0.90 * pu_strength[step - 1]
                + 0.10 * rng.uniform(4, 8)
            )

        # Interference calculation
        pu_interference = (
            occupancy * pu_strength * fading
        )

        traffic_interference = 0.5 + 2.0 * traffic
        noise_power = 1.0

        interference = (
            noise_power
            + pu_interference
            + traffic_interference
        )

        # Preserving the existing SNR assumption
        snr_db = rng.uniform(-5, 20, TIME_SLOTS)

        # Current-to-future samples
        for step in range(1, TIME_SLOTS - 1):

            all_data.append({
                "channel_id": channel + 1,
                "pu_occupancy_previous": occupancy[step - 1],
                "pu_occupancy_current": occupancy[step],
                "traffic_load_previous": traffic[step - 1],
                "traffic_load_current": traffic[step],
                "interference_previous": interference[step - 1],
                "interference_current": interference[step],
                "snr_db_current": snr_db[step],
                "future_interference": interference[step + 1]
            })

    return pd.DataFrame(all_data)


# ------------------------------------------------------------
# Evaluate one controller on a test scenario
# ------------------------------------------------------------

def evaluate_controller(test_data, strategy):

    previous_channel = 1
    decisions = []

    for slot, group in test_data.groupby(
        "test_slot", sort=True
    ):

        group = group.copy()

        if strategy == "Current-based":
            group["score"] = group["interference_current"]
        else:
            group["score"] = group["predicted_interference"]

        # Equal switching penalty for both strategies
        switching_mask = (
            group["channel_id"] != previous_channel
        )

        group.loc[switching_mask, "score"] += (
            SWITCHING_PENALTY
        )

        selected = group.loc[group["score"].idxmin()]

        selected_channel = int(selected["channel_id"])

        switched = (
            selected_channel != previous_channel
        )

        decisions.append({
            "selected_channel": selected_channel,
            "future_interference": float(
                selected["future_interference"]
            ),
            "switched": switched
        })

        previous_channel = selected_channel

    decisions = pd.DataFrame(decisions)

    return {
        "average_future_interference":
            decisions["future_interference"].mean(),
        "channel_switches":
            int(decisions["switched"].sum()),
        "switching_rate_percent":
            decisions["switched"].mean() * 100
    }


# ------------------------------------------------------------
# Run independent training and testing scenarios
# ------------------------------------------------------------

all_results = []

print("\n20-SEED ROBUSTNESS EXPERIMENT")
print("============================")

for seed_number in range(NUM_SEEDS):

    print(
        f"\nRunning seed {seed_number + 1}/{NUM_SEEDS}..."
    )

    # Different seeds for training and testing
    train_seed = 10000 + seed_number
    test_seed = 20000 + seed_number

    train_data = generate_scenario(train_seed)
    test_data = generate_scenario(test_seed)

    X_train = train_data[FEATURES]
    y_train = train_data["future_interference"]

    # Train Random Forest using training scenario only
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42 + seed_number,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predict on independent test scenario
    test_data["predicted_interference"] = model.predict(
        test_data[FEATURES]
    )

    # Align the eight channel observations by time slot
    test_data["test_slot"] = (
        test_data.groupby("channel_id").cumcount()
    )

    # Evaluate both strategies on the exact same test scenario
    current_result = evaluate_controller(
        test_data, "Current-based"
    )

    ai_result = evaluate_controller(
        test_data, "AI-predicted"
    )

    all_results.append({
        "seed": seed_number + 1,

        "current_interference":
            current_result["average_future_interference"],

        "ai_interference":
            ai_result["average_future_interference"],

        "current_switches":
            current_result["channel_switches"],

        "ai_switches":
            ai_result["channel_switches"],

        "current_switching_rate":
            current_result["switching_rate_percent"],

        "ai_switching_rate":
            ai_result["switching_rate_percent"],

        "interference_difference_AI_minus_current":
            (
                ai_result["average_future_interference"]
                - current_result["average_future_interference"]
            ),

        "switch_difference_AI_minus_current":
            (
                ai_result["channel_switches"]
                - current_result["channel_switches"]
            )
    })


# ------------------------------------------------------------
# Save individual seed results
# ------------------------------------------------------------

results = pd.DataFrame(all_results)

results_file = (
    "results/multi_seed_robustness_results.csv"
)

results.to_csv(results_file, index=False)


# ------------------------------------------------------------
# Calculate mean, standard deviation and 95% CI
# ------------------------------------------------------------

def calculate_statistics(values):

    values = np.asarray(values, dtype=float)

    mean_value = np.mean(values)
    std_value = np.std(values, ddof=1)

    standard_error = (
        std_value / np.sqrt(len(values))
    )

    critical_value = t.ppf(
        0.975, df=len(values) - 1
    )

    lower = mean_value - critical_value * standard_error
    upper = mean_value + critical_value * standard_error

    return mean_value, std_value, lower, upper


metrics = {
    "Current-based interference":
        results["current_interference"],

    "AI-predicted interference":
        results["ai_interference"],

    "Current-based switches":
        results["current_switches"],

    "AI-predicted switches":
        results["ai_switches"],

    "Current-based switching rate":
        results["current_switching_rate"],

    "AI-predicted switching rate":
        results["ai_switching_rate"],

    "Interference difference (AI - Current)":
        results["interference_difference_AI_minus_current"],

    "Switch difference (AI - Current)":
        results["switch_difference_AI_minus_current"]
}

summary_rows = []

for metric_name, values in metrics.items():

    mean_value, std_value, lower, upper = (
        calculate_statistics(values)
    )

    summary_rows.append({
        "Metric": metric_name,
        "Mean": mean_value,
        "Standard_Deviation": std_value,
        "95%_CI_Lower": lower,
        "95%_CI_Upper": upper
    })

summary = pd.DataFrame(summary_rows)

summary_file = (
    "results/multi_seed_robustness_summary.csv"
)

summary.to_csv(summary_file, index=False)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n\nPER-SEED RESULTS")
print("================")
print(results.round(4).to_string(index=False))

print("\n\nROBUSTNESS SUMMARY")
print("==================")
print(summary.round(4).to_string(index=False))


# ------------------------------------------------------------
# Paired difference interpretation
# ------------------------------------------------------------

difference = results[
    "interference_difference_AI_minus_current"
]

mean_diff, std_diff, ci_lower, ci_upper = (
    calculate_statistics(difference)
)

print("\n\nPAIRED INTERFERENCE COMPARISON")
print("==============================")

print(f"Mean AI - Current difference: {mean_diff:.4f}")
print(f"Standard deviation: {std_diff:.4f}")
print(f"95% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_upper < 0:
    print(
        "Interpretation: AI has lower average interference "
        "with a 95% confidence interval below zero."
    )
elif ci_lower > 0:
    print(
        "Interpretation: AI has higher average interference "
        "with a 95% confidence interval above zero."
    )
else:
    print(
        "Interpretation: The confidence interval includes zero. "
        "A clear difference is not established."
    )


# ------------------------------------------------------------
# Plot interference and switching comparison
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    results["seed"],
    results["current_interference"],
    marker="o",
    label="Current-based"
)

axes[0].plot(
    results["seed"],
    results["ai_interference"],
    marker="s",
    label="AI-predicted"
)

axes[0].set_xlabel("Independent Seed")
axes[0].set_ylabel("Average Future Interference")
axes[0].set_title("Interference Across Independent Seeds")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    results["seed"],
    results["current_switching_rate"],
    marker="o",
    label="Current-based"
)

axes[1].plot(
    results["seed"],
    results["ai_switching_rate"],
    marker="s",
    label="AI-predicted"
)

axes[1].set_xlabel("Independent Seed")
axes[1].set_ylabel("Switching Rate (%)")
axes[1].set_title("Switching Rate Across Independent Seeds")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

figure_file = (
    "figures/multi_seed_robustness.png"
)

plt.savefig(figure_file, dpi=300)
plt.show()


print("\nEXPERIMENT COMPLETED")
print("====================")
print("Seed results:", results_file)
print("Summary:", summary_file)
print("Figure:", figure_file)
