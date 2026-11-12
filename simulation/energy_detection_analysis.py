
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulation parameters
num_samples = 100
num_trials = 10000
snr_values = [-10, -5, 0, 5, 10, 15, 20]
threshold = 1.2

results = []

for snr_db in snr_values:

    snr_linear = 10 ** (snr_db / 10)

    detections = 0
    false_alarms = 0

    for _ in range(num_trials):

        # H0: Primary user absent
        noise_h0 = np.random.normal(0, 1, num_samples)
        energy_h0 = np.mean(noise_h0 ** 2)

        if energy_h0 > threshold:
            false_alarms += 1

        # H1: Primary user present
        signal = np.random.normal(
            0, np.sqrt(snr_linear), num_samples
        )
        noise_h1 = np.random.normal(0, 1, num_samples)

        received_signal = signal + noise_h1
        energy_h1 = np.mean(received_signal ** 2)

        if energy_h1 > threshold:
            detections += 1

    probability_false_alarm = false_alarms / num_trials
    probability_detection = detections / num_trials

    results.append({
        "SNR_dB": snr_db,
        "Pd": probability_detection,
        "Pfa": probability_false_alarm
    })

    print(
        f"SNR: {snr_db:>3} dB | "
        f"Pd: {probability_detection:.4f} | "
        f"Pfa: {probability_false_alarm:.4f}"
    )

# Save results
df = pd.DataFrame(results)

df.to_csv(
    "results/energy_detection_results.csv",
    index=False
)

# Plot performance
plt.figure(figsize=(9, 5))

plt.plot(
    df["SNR_dB"], df["Pd"],
    marker="o", label="Probability of Detection (Pd)"
)

plt.plot(
    df["SNR_dB"], df["Pfa"],
    marker="s", label="Probability of False Alarm (Pfa)"
)

plt.xlabel("SNR (dB)")
plt.ylabel("Probability")
plt.title("Energy Detection Performance")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/energy_detection_performance.png",
    dpi=300
)

plt.show()

print("\nEnergy detection experiment completed!")
print("Results saved successfully!")
