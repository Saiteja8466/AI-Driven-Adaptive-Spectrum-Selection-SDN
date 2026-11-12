
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

num_channels = 8
time_slots = 3000

all_data = []

for channel in range(num_channels):

    # Markov model for Primary User activity
    occupancy = np.zeros(time_slots, dtype=int)
    occupancy[0] = np.random.randint(0, 2)

    for t in range(1, time_slots):
        if occupancy[t - 1] == 1:
            occupancy[t] = 0 if np.random.rand() < 0.15 else 1
        else:
            occupancy[t] = 1 if np.random.rand() < 0.10 else 0

    # Temporally correlated traffic load
    traffic = np.zeros(time_slots)
    traffic[0] = np.random.uniform(0.1, 0.9)

    for t in range(1, time_slots):
        traffic[t] = (
            0.85 * traffic[t - 1]
            + 0.15 * np.random.uniform(0, 1)
        )

    # Temporally correlated fading (AR(1) model)
    fading_state = np.zeros(time_slots)
    fading_state[0] = np.random.normal()

    correlation = 0.90

    for t in range(1, time_slots):
        fading_state[t] = (
            correlation * fading_state[t - 1]
            + np.sqrt(1 - correlation**2)
            * np.random.normal()
        )

    fading = np.exp(0.25 * fading_state)

    # Slowly varying PU interference strength
    pu_strength = np.zeros(time_slots)
    pu_strength[0] = np.random.uniform(4, 8)

    for t in range(1, time_slots):
        pu_strength[t] = (
            0.90 * pu_strength[t - 1]
            + 0.10 * np.random.uniform(4, 8)
        )

    # Calculate interference
    pu_interference = (
        occupancy * pu_strength * fading
    )

    traffic_interference = 0.5 + 2.0 * traffic

    noise_power = 1.0

    interference = (
        noise_power
        + pu_interference
        + traffic_interference
    )

    snr_db = np.random.uniform(-5, 20, time_slots)

    # Prepare current-to-future prediction samples
    for t in range(1, time_slots - 1):
        all_data.append({
            "channel_id": channel + 1,
            "pu_occupancy_previous": occupancy[t - 1],
            "pu_occupancy_current": occupancy[t],
            "traffic_load_previous": traffic[t - 1],
            "traffic_load_current": traffic[t],
            "interference_previous": interference[t - 1],
            "interference_current": interference[t],
            "snr_db_current": snr_db[t],
            "future_interference": interference[t + 1]
        })

data = pd.DataFrame(all_data)

data.to_csv(
    "data/temporal_interference_dataset.csv",
    index=False
)

print("UPDATED TEMPORAL DATASET GENERATED")
print("----------------------------------")
print("Channels:", num_channels)
print("Time slots per channel:", time_slots)
print("Total samples:", len(data))

print("\nFuture interference statistics:")
print(data["future_interference"].describe())

# Visualize temporal interference
channel_data = data[
    data["channel_id"] == 1
].head(300)

plt.figure(figsize=(10, 5))

plt.plot(
    channel_data["interference_current"].values,
    label="Current Interference"
)

plt.plot(
    channel_data["future_interference"].values,
    label="Future Interference"
)

plt.xlabel("Time Slot")
plt.ylabel("Interference Power")
plt.title("Temporally Correlated Interference - Channel 1")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/temporal_interference_variation.png",
    dpi=300
)

plt.show()

print("\nUpdated dataset and graph saved!")
