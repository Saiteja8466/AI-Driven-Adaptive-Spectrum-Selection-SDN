
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

# Load temporal interference dataset
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

# Chronological train-test split per channel
train_indices = []
test_indices = []

for channel_id, group in data.groupby("channel_id", sort=True):
    indices = group.index.to_list()
    split_point = int(len(indices) * 0.70)

    train_indices.extend(indices[:split_point])
    test_indices.extend(indices[split_point:])

X_train = X.loc[train_indices]
X_test = X.loc[test_indices]

y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

# Train prediction model using training data only
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future interference for test samples
test = data.loc[test_indices].copy()
test["predicted_interference"] = model.predict(X_test)
test["actual_future_interference"] = y_test

# Give each channel's test samples a matching time index
test["test_slot"] = (
    test.groupby("channel_id").cumcount()
)

# Evaluate channel selection strategies
rng = np.random.default_rng(42)

selected_results = []

for slot, group in test.groupby("test_slot", sort=True):

    # Random selection
    random_row = group.iloc[
        rng.integers(0, len(group))
    ]

    # Select channel with lowest current interference
    current_row = group.loc[
        group["interference_current"].idxmin()
    ]

    # Select channel with lowest predicted future interference
    ai_row = group.loc[
        group["predicted_interference"].idxmin()
    ]

    # Oracle: lowest actual future interference
    oracle_row = group.loc[
        group["actual_future_interference"].idxmin()
    ]

    selected_results.append({
        "test_slot": slot,
        "random_channel": random_row["channel_id"],
        "random_future_interference":
            random_row["actual_future_interference"],

        "current_based_channel": current_row["channel_id"],
        "current_based_future_interference":
            current_row["actual_future_interference"],

        "ai_selected_channel": ai_row["channel_id"],
        "ai_selected_future_interference":
            ai_row["actual_future_interference"],

        "oracle_channel": oracle_row["channel_id"],
        "oracle_future_interference":
            oracle_row["actual_future_interference"]
    })

results = pd.DataFrame(selected_results)

print("ADAPTIVE SPECTRUM SELECTION RESULTS")
print("-----------------------------------")
print("Test time slots:", len(results))

print(
    "\nAverage selected future interference:"
)

print(
    "Random selection:",
    round(
        results["random_future_interference"].mean(),
        4
    )
)

print(
    "Current-based selection:",
    round(
        results["current_based_future_interference"].mean(),
        4
    )
)

print(
    "AI predicted selection:",
    round(
        results["ai_selected_future_interference"].mean(),
        4
    )
)

print(
    "Oracle reference:",
    round(
        results["oracle_future_interference"].mean(),
        4
    )
)

# Save results
results.to_csv(
    "results/adaptive_spectrum_selection_results.csv",
    index=False
)

print("\nAdaptive selection results saved successfully!")
