
import numpy as np
import matplotlib.pyplot as plt

# Reproducible simulation
np.random.seed(42)

# Simulation parameters
num_samples = 10000
snr_db = 5

# Generate transmitted signal (BPSK)
bits = np.random.randint(0, 2, num_samples)
transmitted_signal = 2 * bits - 1

# Rayleigh fading channel
real = np.random.normal(0, 1, num_samples)
imag = np.random.normal(0, 1, num_samples)

fading = (real + 1j * imag) / np.sqrt(2)

# Apply fading
faded_signal = transmitted_signal * fading

# Convert SNR from dB to linear scale
snr_linear = 10 ** (snr_db / 10)

# Generate AWGN
noise_power = 1 / snr_linear

noise = np.sqrt(noise_power / 2) * (
    np.random.normal(0, 1, num_samples)
    + 1j * np.random.normal(0, 1, num_samples)
)

# Received signal
received_signal = faded_signal + noise

# Calculate received signal energy
received_energy = np.abs(received_signal) ** 2

print("REALISTIC WIRELESS CHANNEL SIMULATION")
print("-------------------------------------")
print("Modulation: BPSK")
print("Channel: Rayleigh Fading + AWGN")
print("SNR:", snr_db, "dB")
print("Average Received Energy:",
      round(np.mean(received_energy), 4))

# Plot received signal energy
plt.figure(figsize=(10, 5))
plt.plot(received_energy[:300])
plt.title("Received Signal Energy - Rayleigh + AWGN")
plt.xlabel("Sample Index")
plt.ylabel("Received Energy")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/rayleigh_awgn_simulation.png",
    dpi=300
)

plt.show()

print("Simulation completed successfully!")
print("Graph saved in figures folder.")
