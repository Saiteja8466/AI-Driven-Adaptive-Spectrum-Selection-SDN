
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(
    "data/temporal_interference_dataset.csv"
)

features = [
    "channel_id",
    "pu_occupancy_previous",
    "pu_occupancy_current",
    "traffic_load_previous",
    "traffic_load_current",
    "interference_previous",
    "interference_current",
    "snr_db_current"
]

X = data[features]
y = data["future_interference"]

# Chronological train-test split
train_indices = []
test_indices = []

for channel_id, group in data.groupby("channel_id", sort=True):
    indices = group.index.to_list()
    split_point = int(len(indices) * 0.70)

    train_indices.extend(indices[:split_point])
    test_indices.extend(indices[split_point:])

# Train Linear Regression
model = LinearRegression()
model.fit(X.loc[train_indices], y.loc[train_indices])

# Prepare test data
test = data.loc[test_indices].copy()

test["predicted_interference"] = model.predict(
    X.loc[test_indices]
)

test["test_slot"] = (
    test.groupby("channel_id").cumcount()
)

penalties = [0.0, 0.25, 0.5, 1.0, 2.0]

results = []

for strategy in ["Current-based", "AI-predicted"]:

    for penalty in penalties:

        previous_channel = None
        total_interference = 0
        total_switches = 0

        for slot, group in test.groupby(
            "test_slot", sort=True
        ):

            group = group.copy()

            if strategy == "Current-based":
                group["selection_score"] = (
                    group["interference_current"]
                )
            else:
                group["selection_score"] = (
                    group["predicted_interference"]
                )

            # Add switching penalty
            if previous_channel is not None:
                group.loc[
                    group["channel_id"] != previous_channel,
                    "selection_score"
                ] += penalty

            selected = group.loc[
                group["selection_score"].idxmin()
            ]

            selected_channel = int(
                selected["channel_id"]
            )

            if previous_channel is not None:
                if selected_channel != previous_channel:
                    total_switches += 1

            total_interference += selected[
                "future_interference"
            ]

            previous_channel = selected_channel

        total_slots = test["test_slot"].nunique()

        results.append({
            "Strategy": strategy,
            "Penalty": penalty,
            "Average_Interference":
                total_interference / total_slots,
            "Channel_Switches": total_switches,
            "Switching_Rate": (
                total_switches / (total_slots - 1)
            ) * 100
        })

comparison = pd.DataFrame(results)

print("FAIR SWITCHING-AWARE POLICY COMPARISON")
print("--------------------------------------")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "results/fair_switching_policy_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
