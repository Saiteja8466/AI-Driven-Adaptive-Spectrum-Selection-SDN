import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/spectrum_model.pkl")

# Channel measurements
channels = pd.DataFrame([
    [10, 1.0, 1.0, 15.0],
    [5, 1.2, 0.5, 2.0],
    [20, 1.0, 2.0, 105.0],
    [8, 1.5, 0.3, 3.0],
    [15, 1.1, 1.0, 40.0]
], columns=[
    "snr", "noise_power", "interference", "energy"
])

# Predict occupancy
predictions = model.predict(channels)

# Find available channels
available = []

for i, prediction in enumerate(predictions):

    if prediction == 0:
        available.append(i)

# Select lowest-energy available channel
if available:

    best = min(
        available,
        key=lambda i: channels.iloc[i]["energy"]
    )

    print("AI-Based Dynamic Spectrum Access")
    print("--------------------------------")
    print("Available Channels:", [i + 1 for i in available])
    print("Selected Best Channel:", best + 1)
    print("Channel Energy:", channels.iloc[best]["energy"])

else:
    print("No available channel found.")