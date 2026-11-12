
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

# ============================================================
# AI-DRIVEN ADAPTIVE CHANNEL SELECTION
# 8-Channel Wireless Network Simulation
# ============================================================

DATA_FILE = "data/temporal_interference_dataset.csv"
MODEL_FILE = "models/temporal_interference_model.pkl"

OUTPUT_FILE = "results/ai_adaptive_channel_selection.csv"

SWITCHING_PENALTY = 1.0
PREVIOUS_CHANNEL = 1

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

# ------------------------------------------------------------
# Load existing temporal dataset
# ------------------------------------------------------------

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

data = pd.read_csv(DATA_FILE)

print("\nAI ADAPTIVE CHANNEL SELECTION")
print("=============================")

# ------------------------------------------------------------
# Chronological training split
# ------------------------------------------------------------

train_indices = []
test_indices = []

for channel_id, group in data.groupby(
    "channel_id", sort=True
):
    indices = group.index.to_list()

    split_point = int(len(indices) * 0.70)

    train_indices.extend(indices[:split_point])
    test_indices.extend(indices[split_point:])

train_data = data.loc[train_indices].copy()
test_data = data.loc[test_indices].copy()

X_train = train_data[FEATURES]
y_train = train_data["future_interference"]

# ------------------------------------------------------------
# Train Random Forest model
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    MODEL_FILE
)

print("\nAI model trained successfully.")

# ------------------------------------------------------------
# Select one common test time slot
# ------------------------------------------------------------

test_data["test_slot"] = (
    test_data.groupby("channel_id").cumcount()
)

# Use a middle test slot for demonstration
selected_test_slot = (
    test_data["test_slot"].nunique() // 2
)

current_state = test_data[
    test_data["test_slot"] == selected_test_slot
].copy()

# ------------------------------------------------------------
# Predict future interference for all channels
# ------------------------------------------------------------

current_state["predicted_interference"] = (
    model.predict(current_state[FEATURES])
)

# ------------------------------------------------------------
# Apply switching penalty
# ------------------------------------------------------------

current_state["switching_cost"] = np.where(
    current_state["channel_id"] != PREVIOUS_CHANNEL,
    SWITCHING_PENALTY,
    0.0
)

current_state["selection_score"] = (
    current_state["predicted_interference"]
    + current_state["switching_cost"]
)

# ------------------------------------------------------------
# Select best channel
# ------------------------------------------------------------

best_index = current_state[
    "selection_score"
].idxmin()

best_channel = current_state.loc[best_index]

# ------------------------------------------------------------
# Display all channel predictions
# ------------------------------------------------------------

display_columns = [
    "channel_id",
    "interference_current",
    "predicted_interference",
    "switching_cost",
    "selection_score",
    "future_interference"
]

print("\nCHANNEL-WISE AI PREDICTIONS")
print("===========================")

print(
    current_state[display_columns]
    .sort_values("selection_score")
    .round(4)
    .to_string(index=False)
)

# ------------------------------------------------------------
# Display selected channel
# ------------------------------------------------------------

print("\nAI CHANNEL SELECTION RESULT")
print("===========================")

print("Previous Channel:", PREVIOUS_CHANNEL)

print(
    "Selected Channel:",
    int(best_channel["channel_id"])
)

print(
    "Predicted Interference:",
    round(
        float(best_channel["predicted_interference"]),
        4
    )
)

print(
    "Switching Cost:",
    round(
        float(best_channel["switching_cost"]),
        4
    )
)

print(
    "Selection Score:",
    round(
        float(best_channel["selection_score"]),
        4
    )
)

print(
    "Actual Future Interference:",
    round(
        float(best_channel["future_interference"]),
        4
    )
)

# ------------------------------------------------------------
# Save complete channel-wise results
# ------------------------------------------------------------

current_state.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nChannel selection results saved:")
print(OUTPUT_FILE)

print("\nDEMONSTRATION COMPLETED")
