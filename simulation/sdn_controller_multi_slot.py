
import pandas as pd
import joblib

# Load temporal dataset
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

# Load trained controller model
model = joblib.load(
    "models/sdn_interference_model.pkl"
)

# Prepare test data chronologically
test_indices = []

for channel_id, group in data.groupby("channel_id", sort=True):
    indices = group.index.to_list()
    split_point = int(len(indices) * 0.70)
    test_indices.extend(indices[split_point:])

test = data.loc[test_indices].copy()

test["predicted_interference"] = model.predict(
    test[features]
)

test["test_slot"] = (
    test.groupby("channel_id").cumcount()
)

# Controller settings
switching_penalty = 0.5
previous_channel = 1

decisions = []

# Run controller for multiple time slots
for slot, group in test.groupby("test_slot", sort=True):

    group = group.copy()

    group["selection_score"] = (
        group["predicted_interference"]
    )

    group.loc[
        group["channel_id"] != previous_channel,
        "selection_score"
    ] += switching_penalty

    selected = group.loc[
        group["selection_score"].idxmin()
    ]

    selected_channel = int(selected["channel_id"])

    switched = selected_channel != previous_channel

    decisions.append({
        "time_slot": slot,
        "previous_channel": previous_channel,
        "selected_channel": selected_channel,
        "predicted_interference":
            selected["predicted_interference"],
        "actual_future_interference":
            selected["future_interference"],
        "channel_switched": switched
    })

    previous_channel = selected_channel

# Save controller logs
results = pd.DataFrame(decisions)

results.to_csv(
    "results/sdn_controller_decisions.csv",
    index=False
)

print("MULTI-SLOT SDN CONTROLLER RESULTS")
print("---------------------------------")
print("Total time slots:", len(results))
print("Channel switches:",
      int(results["channel_switched"].sum()))

print("Switching rate:",
      round(results["channel_switched"].mean() * 100, 2), "%")

print("Average predicted interference:",
      round(results["predicted_interference"].mean(), 4))

print("Average actual future interference:",
      round(results["actual_future_interference"].mean(), 4))

print("\nFirst 10 controller decisions:")
print(results.head(10).to_string(index=False))

print("\nController decision log saved successfully!")
