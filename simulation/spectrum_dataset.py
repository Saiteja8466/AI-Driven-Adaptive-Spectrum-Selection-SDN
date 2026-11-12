import numpy as np
import pandas as pd

np.random.seed(42)

samples = 5000

snr = np.random.uniform(-10, 30, samples)
noise_power = np.random.uniform(0.5, 2.0, samples)
interference = np.random.uniform(0, 5, samples)

occupancy = np.random.randint(0, 2, samples)

signal_power = np.where(
    occupancy == 1,
    10 ** (snr / 10) * noise_power,
    0
)

energy = signal_power + noise_power + interference

data = pd.DataFrame({
    "snr": snr,
    "noise_power": noise_power,
    "interference": interference,
    "energy": energy,
    "occupancy": occupancy
})

data.to_csv("data/spectrum_data.csv", index=False)

print("Dataset generated successfully!")
print(data.head())
print("\nDataset shape:", data.shape)