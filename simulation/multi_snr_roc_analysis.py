
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

np.random.seed(42)

num_samples = 100000
snr_values = [-10, -5, 0, 5, 10]

plt.figure(figsize=(9, 6))

for snr_db in snr_values:

    snr_linear = 10 ** (snr_db / 10)

    # H0: Noise only
    noise_h0 = np.random.normal(0, 1, num_samples)
    energy_h0 = noise_h0 ** 2

    # H1: Signal plus noise
    signal = np.random.normal(
        0, np.sqrt(snr_linear), num_samples
    )
    noise_h1 = np.random.normal(0, 1, num_samples)

    received = signal + noise_h1
    energy_h1 = received ** 2

    energies = np.concatenate([energy_h0, energy_h1])

    labels = np.concatenate([
        np.zeros(num_samples),
        np.ones(num_samples)
    ])

    fpr, tpr, thresholds = roc_curve(labels, energies)

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"SNR {snr_db} dB (AUC={roc_auc:.4f})"
    )

    print(f"SNR: {snr_db:>3} dB | AUC: {roc_auc:.4f}")

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")

plt.xlabel("Probability of False Alarm")
plt.ylabel("Probability of Detection")
plt.title("ROC Comparison at Different SNR Levels")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("figures/multi_snr_roc.png", dpi=300)
plt.show()

print("\nMulti-SNR ROC analysis completed!")
