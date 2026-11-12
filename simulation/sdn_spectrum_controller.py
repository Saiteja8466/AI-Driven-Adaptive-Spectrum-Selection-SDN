
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression

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

# Chronological training split
train_indices = []

for channel_id, group in data.groupby("channel_id", sort=True):
    indices = group.index.to_list()
    split_point = int(len(indices) * 0.70)
    train_indices.extend(indices[:split_point])

# Train prediction model
model = LinearRegression()

model.fit(
    data.loc[train_indices, features],
    data.loc[train_indices, "future_interference"]
)

# Save model
joblib.dump(
    model,
    "models/sdn_interference_model.pkl"
)

print("SDN-INSPIRED CENTRALIZED CONTROLLER")
print("-----------------------------------")

# Simulate measurements from 8 channels
measurements = data.groupby("channel_id").tail(1).copy()

# Predict future interference
measurements["predicted_interference"] = model.predict(
    measurements[features]
)

# Switching cost
switching_penalty = 0.5

# Assume currently connected to Channel 1
previous_channel = 1

measurements["selection_score"] = (
    measurements["predicted_interference"]
)

measurements.loc[
    measurements["channel_id"] != previous_channel,
    "selection_score"
] += switching_penalty

# Select best channel
selected = measurements.loc[
    measurements["selection_score"].idxmin()
]

print("\nChannel measurements:")
print(
    measurements[
        [
            "channel_id",
            "interference_current",
            "predicted_interference",
            "selection_score"
        ]
    ].to_string(index=False)
)

print("\nController decision:")
print("Previous channel:", previous_channel)
print("Selected channel:", int(selected["channel_id"]))
print(
    "Predicted interference:",
    round(selected["predicted_interference"], 4)
)
print(
    "Selection score:",
    round(selected["selection_score"], 4)
)

if int(selected["channel_id"]) != previous_channel:
    print("Action: SWITCH CHANNEL")
else:
    print("Action: MAINTAIN CURRENT CHANNEL")

print("\nController execution completed!")
