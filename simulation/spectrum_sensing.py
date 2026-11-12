import numpy as np
import matplotlib.pyplot as plt

# Simulate a wireless spectrum signal
np.random.seed(42)

num_samples = 1000
noise = np.random.normal(0, 1, num_samples)

# Primary user signal
signal = np.zeros(num_samples)
signal[300:700] = 3

# Received signal
received_signal = signal + noise

# Energy detection
energy = np.mean(received_signal ** 2)

threshold = 2.0

if energy > threshold:
    print("Primary User Detected: Channel is BUSY")
else:
    print("Primary User Not Detected: Channel is AVAILABLE")

print("Measured Signal Energy:", round(energy, 4))

# Plot the received signal
plt.figure(figsize=(10, 4))
plt.plot(received_signal)
plt.title("Cognitive Radio - Spectrum Sensing")
plt.xlabel("Sample Index")
plt.ylabel("Signal Amplitude")
plt.grid(True)
plt.show()