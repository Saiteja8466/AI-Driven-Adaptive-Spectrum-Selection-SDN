
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load temporal dataset
data = pd.read_csv("data/temporal_interference_dataset.csv")

# Input features
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

# Chronological split separately for each channel
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

# Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict future interference
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Persistence baseline: assume next interference equals current
baseline_predictions = X_test["interference_current"]

baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print("TEMPORAL INTERFERENCE PREDICTION RESULTS")
print("----------------------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(f"\nModel MAE: {mae:.4f}")
print(f"Model RMSE: {rmse:.4f}")
print(f"Model R2 Score: {r2:.4f}")

print(f"\nPersistence Baseline MAE: {baseline_mae:.4f}")

# Save trained model
joblib.dump(
    model,
    "models/temporal_interference_model.pkl"
)

# Save actual and predicted values
results = pd.DataFrame({
    "actual_interference": y_test,
    "predicted_interference": predictions,
    "current_interference": X_test["interference_current"]
})

results.to_csv(
    "results/temporal_interference_predictions.csv",
    index=False
)

# Plot actual vs predicted interference
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Ideal Prediction"
)

plt.xlabel("Actual Future Interference")
plt.ylabel("Predicted Future Interference")
plt.title("Actual vs Predicted Interference")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/temporal_interference_prediction.png",
    dpi=300
)

plt.show()

print("\nModel, predictions and graph saved successfully!")
