import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

num_channels = 5
num_samples = 1000
threshold = 2.0

channel_energies = []

for channel in range(num_channels):

    noise = np.random.normal(0, 1, num_samples)

    # Simulate different channel occupancy
    if channel in [1, 3]:
        signal = np.ones(num_samples) * 2.5
    else:
        signal = np.zeros(num_samples)

    received_signal = signal + noise

    energy = np.mean(received_signal ** 2)
    channel_energies.append(energy)

    if energy > threshold:
        status = "BUSY"
    else:
        status = "AVAILABLE"

    print(
        f"Channel {channel + 1}: "
        f"Energy = {energy:.4f}, Status = {status}"
    )

# Plot channel energy
plt.figure(figsize=(9, 5))

plt.bar(
    range(1, num_channels + 1),
    channel_energies
)

plt.axhline(
    y=threshold,
    linestyle="--",
    label="Detection Threshold"
)

plt.xlabel("Channel Number")
plt.ylabel("Signal Energy")
plt.title("Multi-Channel Spectrum Sensing")

plt.legend()
plt.grid(axis="y")
plt.show()