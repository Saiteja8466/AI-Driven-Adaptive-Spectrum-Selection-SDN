
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

num_samples = 100000
snr_values = [-10, -5, 0, 5, 10, 15, 20]

results = []

# Generate BPSK signal
bits = np.random.randint(0, 2, num_samples)
transmitted_signal = 2 * bits - 1

# Rayleigh fading
fading = (
    np.random.normal(0, 1, num_samples)
    + 1j * np.random.normal(0, 1, num_samples)
) / np.sqrt(2)

faded_signal = transmitted_signal * fading

for snr_db in snr_values:

    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.normal(0, 1, num_samples)
        + 1j * np.random.normal(0, 1, num_samples)
    )

    received_signal = faded_signal + noise

    # Coherent detection with perfect channel knowledge
    detected_signal = np.real(
        received_signal * np.conj(fading)
    )

    detected_bits = (detected_signal >= 0).astype(int)

    ber = np.mean(detected_bits != bits)

    results.append({
        "SNR_dB": snr_db,
        "BER": ber
    })

    print(f"SNR: {snr_db:>3} dB | BER: {ber:.6f}")

# Save results
df = pd.DataFrame(results)
df.to_csv("results/snr_ber_results.csv", index=False)

# Plot BER
plt.figure(figsize=(9, 5))
plt.semilogy(df["SNR_dB"], df["BER"], marker="o")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BPSK Performance over Rayleigh Fading + AWGN")
plt.grid(True, which="both")
plt.tight_layout()

plt.savefig("figures/snr_ber_performance.png", dpi=300)
plt.show()

print("\nSNR performance analysis completed!")
print("Results saved in results folder.")
