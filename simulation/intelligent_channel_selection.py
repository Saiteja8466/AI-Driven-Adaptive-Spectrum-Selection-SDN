import numpy as np

np.random.seed(42)

# Channel information
channel_energies = np.array([
    0.9583,
    7.6031,
    0.9662,
    7.2107,
    0.9863
])

threshold = 2.0

# Identify available channels
available_channels = []

for i, energy in enumerate(channel_energies):

    if energy < threshold:
        available_channels.append(i)

# Select the channel with minimum energy
if available_channels:

    best_channel = min(
        available_channels,
        key=lambda i: channel_energies[i]
    )

    print("Available Channels:")

    for channel in available_channels:
        print(f"Channel {channel + 1}")

    print("\nIntelligent Channel Selection Result")
    print("Selected Channel:", best_channel + 1)
    print("Selected Channel Energy:",
          channel_energies[best_channel])

else:
    print("No available channels found.")