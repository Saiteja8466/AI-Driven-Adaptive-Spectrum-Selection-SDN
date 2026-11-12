
import pandas as pd
import joblib

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

# Load trained AI model
model = joblib.load(
    "models/sdn_interference_model.pkl"
)

# Chronological test split
test_indices = []

for channel_id, group in data.groupby("channel_id", sort=True):
    indices = group.index.to_list()
    split_point = int(len(indices) * 0.70)
    test_indices.extend(indices[split_point:])

test = data.loc[test_indices].copy()

# AI future interference prediction
test["predicted_interference"] = model.predict(
    test[features]
)

# Align channel observations by test time slot
test["test_slot"] = (
    test.groupby("channel_id").cumcount()
)

switching_penalty = 0.5

strategies = [
    "Current-based",
    "AI-predicted"
]

all_results = []

for strategy in strategies:

    previous_channel = 1
    decisions = []

    for slot, group in test.groupby(
        "test_slot", sort=True
    ):

        group = group.copy()

        if strategy == "Current-based":
            group["score"] = group[
                "interference_current"
            ]
        else:
            group["score"] = group[
                "predicted_interference"
            ]

        # Apply equal switching cost
        group.loc[
            group["channel_id"] != previous_channel,
            "score"
        ] += switching_penalty

        selected = group.loc[
            group["score"].idxmin()
        ]

        selected_channel = int(
            selected["channel_id"]
        )

        switched = (
            selected_channel != previous_channel
        )

        decisions.append({
            "time_slot": slot,
            "selected_channel": selected_channel,
            "future_interference":
                selected["future_interference"],
            "switched": switched
        })

        previous_channel = selected_channel

    strategy_results = pd.DataFrame(decisions)

    all_results.append({
        "Strategy": strategy,
        "Average_Future_Interference":
            strategy_results["future_interference"].mean(),
        "Channel_Switches":
            strategy_results["switched"].sum(),
        "Switching_Rate_Percent":
            strategy_results["switched"].mean() * 100
    })

comparison = pd.DataFrame(all_results)

print("CENTRALIZED CONTROLLER COMPARISON")
print("---------------------------------")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "results/sdn_controller_comparison.csv",
    index=False
)

print("\nController comparison saved successfully!")
