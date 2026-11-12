
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load saved predictions
data = pd.read_csv(
    "results/temporal_interference_predictions.csv"
)

# Calculate prediction errors
data["absolute_error"] = abs(
    data["actual_interference"]
    - data["predicted_interference"]
)

data["baseline_error"] = abs(
    data["actual_interference"]
    - data["current_interference"]
)

print("PREDICTION ERROR ANALYSIS")
print("-------------------------")

print("\nAI Model:")
print("MAE:", round(data["absolute_error"].mean(), 4))
print("RMSE:", round(
    np.sqrt(np.mean(
        (data["actual_interference"]
         - data["predicted_interference"]) ** 2
    )), 4
))

print("\nPersistence Baseline:")
print("MAE:", round(data["baseline_error"].mean(), 4))

# Analyse high-interference predictions
high = data[data["actual_interference"] > 8]

print("\nHigh Interference Analysis:")
print("High interference samples:", len(high))

if len(high) > 0:
    print(
        "AI high-interference MAE:",
        round(high["absolute_error"].mean(), 4)
    )

# Plot error comparison
plt.figure(figsize=(8, 5))

plt.hist(
    data["absolute_error"],
    bins=40,
    alpha=0.7,
    label="AI Model Error"
)

plt.hist(
    data["baseline_error"],
    bins=40,
    alpha=0.5,
    label="Baseline Error"
)

plt.xlabel("Absolute Prediction Error")
plt.ylabel("Frequency")
plt.title("AI vs Persistence Baseline Error Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/prediction_error_analysis.png",
    dpi=300
)

plt.show()

print("\nAnalysis graph saved successfully!")
