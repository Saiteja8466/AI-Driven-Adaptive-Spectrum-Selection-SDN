import joblib
import pandas as pd

# Load trained Random Forest model
model = joblib.load("models/spectrum_model.pkl")

# New channel measurements
test_channels = pd.DataFrame([
    [10, 1.0, 1.0, 15.0],
    [5, 1.2, 0.5, 2.0],
    [20, 1.0, 2.0, 105.0],
    [8, 1.5, 0.3, 3.0],
    [15, 1.1, 1.0, 40.0]
], columns=[
    "snr", "noise_power", "interference", "energy"
])

# Predict channel occupancy
predictions = model.predict(test_channels)

print("AI-Based Spectrum Occupancy Prediction")
print("--------------------------------------")

available_channels = []

for i, prediction in enumerate(predictions):

    if prediction == 0:
        status = "AVAILABLE"
        available_channels.append(i + 1)
    else:
        status = "OCCUPIED"

    print(f"Channel {i + 1}: {status}")

print("\nAvailable Channels:", available_channels)

if available_channels:
    print("AI identified available channels successfully.")
else:
    print("No available channels detected.")