
import pandas as pd

# Load adaptive selection results
data = pd.read_csv(
    "results/adaptive_spectrum_selection_results.csv"
)

strategies = {
    "Random": "random_channel",
    "Current-based": "current_based_channel",
    "AI predicted": "ai_selected_channel",
    "Oracle": "oracle_channel"
}

print("CHANNEL SWITCHING ANALYSIS")
print("---------------------------")

for name, column in strategies.items():

    switches = (
        data[column].diff().fillna(0) != 0
    ).sum()

    total_slots = len(data) - 1

    switching_rate = (
        switches / total_slots
    ) * 100

    print(f"\n{name} strategy")
    print("Channel switches:", int(switches))
    print(f"Switching rate: {switching_rate:.2f}%")

print("\nAnalysis completed!")
