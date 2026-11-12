
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

np.random.seed(42)

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

# Chronological split per channel
train_indices = []
test_indices = []

for channel_id, group in data.groupby("channel_id", sort=True):
    indices = group.index.to_list()
    split_point = int(len(indices) * 0.70)

    train_indices.extend(indices[:split_point])
    test_indices.extend(indices[split_point:])

# Train prediction model
model = LinearRegression()
model.fit(X.loc[train_indices], y.loc[train_indices])

# Prepare test observations
test = data.loc[test_indices].copy()

test["predicted"] = model.predict(
    X.loc[test_indices]
)

test["test_slot"] = (
    test.groupby("channel_id").cumcount()
)

# Evaluate two switching penalties
penalties = [0.5, 1.0]

for penalty in penalties:

    current_values = []
    ai_values = []

    previous_current = None
    previous_ai = None

    for slot, group in test.groupby("test_slot", sort=True):

        group = group.copy()

        # Current-based policy
        current_score = group["interference_current"].copy()

        if previous_current is not None:
            current_score = current_score + (
                group["channel_id"] != previous_current
            ) * penalty

        current_row = group.loc[current_score.idxmin()]
        previous_current = int(current_row["channel_id"])

        # AI-predicted policy
        ai_score = group["predicted"].copy()

        if previous_ai is not None:
            ai_score = ai_score + (
                group["channel_id"] != previous_ai
            ) * penalty

        ai_row = group.loc[ai_score.idxmin()]
        previous_ai = int(ai_row["channel_id"])

        current_values.append(
            current_row["future_interference"]
        )

        ai_values.append(
            ai_row["future_interference"]
        )

    current_values = np.array(current_values)
    ai_values = np.array(ai_values)

    # Paired difference: negative favours AI
    differences = ai_values - current_values

    # Moving block bootstrap
    block_length = 20
    num_samples = len(differences)
    repetitions = 2000

    bootstrap_means = []

    for _ in range(repetitions):

        sampled = []

        while len(sampled) < num_samples:
            start = np.random.randint(
                0, num_samples - block_length + 1
            )

            sampled.extend(
                differences[start:start + block_length]
            )

        sampled = np.array(sampled[:num_samples])
        bootstrap_means.append(sampled.mean())

    lower, upper = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    print("\nSwitching penalty:", penalty)
    print(
        "Current-based average:",
        round(current_values.mean(), 4)
    )
    print(
        "AI-predicted average:",
        round(ai_values.mean(), 4)
    )
    print(
        "Mean paired difference (AI - Current):",
        round(differences.mean(), 4)
    )
    print(
        "95% block-bootstrap CI:",
        (round(lower, 4), round(upper, 4))
    )

    if upper < 0:
        print("Interpretation: AI advantage supported by this CI.")
    elif lower > 0:
        print("Interpretation: Current-based strategy is better.")
    else:
        print("Interpretation: No clear difference established.")

print("\nStatistical validation completed!")
