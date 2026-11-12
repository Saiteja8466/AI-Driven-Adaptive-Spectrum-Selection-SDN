
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load(
    "models/temporal_interference_model.pkl"
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

# Extract feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("TEMPORAL MODEL FEATURE IMPORTANCE")
print("---------------------------------")
print(importance.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(9, 5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Input Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "figures/temporal_feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance graph saved!")
