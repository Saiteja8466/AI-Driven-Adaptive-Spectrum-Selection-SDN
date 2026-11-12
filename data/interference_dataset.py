
import numpy as np
import pandas as pd

np.random.seed(42)

# Simulation settings
num_samples = 10000

# Wireless channel conditions
snr_db = np.random.uniform(-10, 25, num_samples)

noise_power = np.random.uniform(0.5, 2.0, num_samples)

traffic_load = np.random.uniform(0, 1, num_samples)

num_interferers = np.random.randint(0, 6, num_samples)

# Interference power increases with interferer count
interference_power = (
    num_interferers * np.random.uniform(0.1, 1.0, num_samples)
    * (0.5 + traffic_load)
)

# Total received power
signal_power = noise_power * (10 ** (snr_db / 10))

received_power = (
    signal_power + noise_power + interference_power
)

# Interference category
interference_level = np.where(
    interference_power < 1.0, 0,
    np.where(interference_power < 3.0, 1, 2)
)

# Create dataset
data = pd.DataFrame({
    "snr_db": snr_db,
    "noise_power": noise_power,
    "traffic_load": traffic_load,
    "num_interferers": num_interferers,
    "received_power": received_power,
    "interference_power": interference_power,
    "interference_level": interference_level
})

# Save dataset
data.to_csv(
    "data/interference_dataset.csv",
    index=False
)

print("INTERFERENCE DATASET GENERATED")
print("--------------------------------")
print("Total samples:", len(data))
print("\nFirst five samples:")
print(data.head())

print("\nInterference level distribution:")
print(data["interference_level"].value_counts().sort_index())

print("\nDataset saved successfully!")
