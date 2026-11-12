
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Chronological split for each channel
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

# Model 1: Persistence baseline
baseline_pred = X_test["interference_current"]

# Model 2: Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Model 3: Linear Regression
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Evaluation function
def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }

results = [
    evaluate("Persistence", y_test, baseline_pred),
    evaluate("Random Forest", y_test, rf_pred),
    evaluate("Linear Regression", y_test, lr_pred)
]

comparison = pd.DataFrame(results)

print("INTERFERENCE MODEL COMPARISON")
print("--------------------------------")
print(comparison.to_string(index=False))

comparison.to_csv(
    "results/interference_model_comparison.csv",
    index=False
)

print("\nComparison results saved successfully!")
