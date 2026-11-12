
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# ============================================================
# SWITCHING PENALTY OPTIMIZATION
# 20 Independent Seeds
# ============================================================

NUM_SEEDS = 20
NUM_CHANNELS = 8
TIME_SLOTS = 3000

PENALTIES = [0, 0.25, 0.5, 1, 2, 4, 8]

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
# Generate synthetic wireless scenario
# ------------------------------------------------------------

def generate_scenario(seed):

    rng = np.random.default_rng(seed)
    all_data = []

    for channel in range(NUM_CHANNELS):

        # Primary User Markov activity
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

        # Correlated traffic
        traffic = np.zeros(TIME_SLOTS)
        traffic[0] = rng.uniform(0.1, 0.9)

        for step in range(1, TIME_SLOTS):
            traffic[step] = (
                0.85 * traffic[step - 1]
                + 0.15 * rng.uniform(0, 1)
            )

        # Correlated fading
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

        # Primary User interference strength
        pu_strength = np.zeros(TIME_SLOTS)
        pu_strength[0] = rng.uniform(4, 8)

        for step in range(1, TIME_SLOTS):
            pu_strength[step] = (
                0.90 * pu_strength[step - 1]
                + 0.10 * rng.uniform(4, 8)
            )

        # Interference calculation
        pu_interference = occupancy * pu_strength * fading

        traffic_interference = 0.5 + 2.0 * traffic

        interference = (
            1.0 + pu_interference + traffic_interference
        )

        snr_db = rng.uniform(-5, 20, TIME_SLOTS)

        # Current-to-future dataset
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
# Evaluate controller
# ------------------------------------------------------------

def evaluate_controller(test_data, strategy, penalty):

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

        # Switching cost
        switching_mask = (
            group["channel_id"] != previous_channel
        )

        group.loc[switching_mask, "score"] += penalty

        selected = group.loc[group["score"].idxmin()]

        selected_channel = int(selected["channel_id"])

        switched = selected_channel != previous_channel

        decisions.append({
            "future_interference":
                float(selected["future_interference"]),
            "switched": switched
        })

        previous_channel = selected_channel

    decisions = pd.DataFrame(decisions)

    return {
        "average_interference":
            decisions["future_interference"].mean(),

        "channel_switches":
            int(decisions["switched"].sum()),

        "switching_rate":
            decisions["switched"].mean() * 100
    }


# ------------------------------------------------------------
# Run experiments
# ------------------------------------------------------------

all_results = []

print("\nSWITCHING PENALTY OPTIMIZATION")
print("==============================")

for seed_number in range(NUM_SEEDS):

    print(
        f"\nRunning seed {seed_number + 1}/{NUM_SEEDS}"
    )

    train_data = generate_scenario(10000 + seed_number)
    test_data = generate_scenario(20000 + seed_number)

    # Train AI model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42 + seed_number,
        n_jobs=-1
    )

    model.fit(
        train_data[FEATURES],
        train_data["future_interference"]
    )

    # Predict independent test scenario
    test_data["predicted_interference"] = model.predict(
        test_data[FEATURES]
    )

    # Align channels by time slot
    test_data["test_slot"] = (
        test_data.groupby("channel_id").cumcount()
    )

    for penalty in PENALTIES:

        for strategy in ["Current-based", "AI-predicted"]:

            result = evaluate_controller(
                test_data,
                strategy,
                penalty
            )

            all_results.append({
                "seed": seed_number + 1,
                "strategy": strategy,
                "penalty": penalty,
                "average_interference":
                    result["average_interference"],
                "channel_switches":
                    result["channel_switches"],
                "switching_rate_percent":
                    result["switching_rate"]
            })


# ------------------------------------------------------------
# Save individual results
# ------------------------------------------------------------

results = pd.DataFrame(all_results)

results.to_csv(
    "results/switching_penalty_seed_results.csv",
    index=False
)


# ------------------------------------------------------------
# Aggregate results
# ------------------------------------------------------------

summary = (
    results
    .groupby(["strategy", "penalty"])
    .agg(
        mean_interference=("average_interference", "mean"),
        std_interference=("average_interference", "std"),
        mean_switches=("channel_switches", "mean"),
        std_switches=("channel_switches", "std"),
        mean_switching_rate=("switching_rate_percent", "mean"),
        std_switching_rate=("switching_rate_percent", "std")
    )
    .reset_index()
)

summary.to_csv(
    "results/switching_penalty_summary.csv",
    index=False
)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n\nSWITCHING PENALTY SUMMARY")
print("=========================")

print(summary.round(4).to_string(index=False))


# ------------------------------------------------------------
# Graph 1: Interference vs penalty
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for strategy in ["Current-based", "AI-predicted"]:

    subset = summary[
        summary["strategy"] == strategy
    ]

    plt.plot(
        subset["penalty"],
        subset["mean_interference"],
        marker="o",
        label=strategy
    )

plt.xlabel("Switching Penalty")
plt.ylabel("Average Future Interference")
plt.title("Interference vs Switching Penalty")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/penalty_interference_tradeoff.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Graph 2: Switching rate vs penalty
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for strategy in ["Current-based", "AI-predicted"]:

    subset = summary[
        summary["strategy"] == strategy
    ]

    plt.plot(
        subset["penalty"],
        subset["mean_switching_rate"],
        marker="o",
        label=strategy
    )

plt.xlabel("Switching Penalty")
plt.ylabel("Average Switching Rate (%)")
plt.title("Switching Rate vs Switching Penalty")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/penalty_switching_tradeoff.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Graph 3: Interference-switching trade-off
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for strategy in ["Current-based", "AI-predicted"]:

    subset = summary[
        summary["strategy"] == strategy
    ]

    plt.plot(
        subset["mean_switching_rate"],
        subset["mean_interference"],
        marker="o",
        label=strategy
    )

plt.xlabel("Average Switching Rate (%)")
plt.ylabel("Average Future Interference")
plt.title("Interference-Switching Trade-off")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/interference_switching_pareto.png",
    dpi=300
)

plt.show()


print("\nEXPERIMENT COMPLETED")
print("====================")
print("results/switching_penalty_seed_results.csv")
print("results/switching_penalty_summary.csv")
print("figures/penalty_interference_tradeoff.png")
print("figures/penalty_switching_tradeoff.png")
print("figures/interference_switching_pareto.png")
