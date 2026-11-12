
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "results/fair_switching_policy_comparison.csv"
)

plt.figure(figsize=(9, 6))

for strategy in data["Strategy"].unique():

    subset = data[data["Strategy"] == strategy]

    plt.plot(
        subset["Switching_Rate"],
        subset["Average_Interference"],
        marker="o",
        label=strategy
    )

plt.xlabel("Channel Switching Rate (%)")
plt.ylabel("Average Future Interference")
plt.title("Interference vs Channel Switching Trade-off")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/switching_tradeoff.png",
    dpi=300
)

plt.show()

print("Trade-off visualization saved successfully!")
