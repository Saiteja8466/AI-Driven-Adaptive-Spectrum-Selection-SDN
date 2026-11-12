
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

np.random.seed(42)

# Simulation parameters
num_samples = 100000
snr_db = 0

snr_linear = 10 ** (snr_db / 10)

# H0: Primary user absent
noise_h0 = np.random.normal(0, 1, num_samples)
energy_h0 = noise_h0 ** 2

# H1: Primary user present
signal = np.random.normal(
    0, np.sqrt(snr_linear), num_samples
)
noise_h1 = np.random.normal(0, 1, num_samples)

received_signal = signal + noise_h1
energy_h1 = received_signal ** 2

# Combine observations
energies = np.concatenate([energy_h0, energy_h1])

labels = np.concatenate([
    np.zeros(num_samples),
    np.ones(num_samples)
])

# Calculate ROC using sklearn
fpr, tpr, thresholds = roc_curve(labels, energies)

# Calculate AUC
roc_auc = auc(fpr, tpr)

print("ROC CURVE ANALYSIS")
print("------------------")
print("SNR:", snr_db, "dB")
print("AUC:", round(roc_auc, 4))

# Plot ROC curve
plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Energy Detector (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("Probability of False Alarm")
plt.ylabel("Probability of Detection")
plt.title("ROC Curve - Energy Detection")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("figures/roc_curve.png", dpi=300)
plt.show()

print("ROC curve saved successfully!")
