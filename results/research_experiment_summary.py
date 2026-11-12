
import pandas as pd
from pathlib import Path

# Project result files
files = {
    "Prediction Model Comparison":
        "results/interference_model_comparison.csv",

    "Adaptive Spectrum Selection":
        "results/adaptive_spectrum_selection_results.csv",

    "Switching Aware Comparison":
        "results/fair_switching_policy_comparison.csv",

    "SDN Controller Comparison":
        "results/sdn_controller_comparison.csv"
}

print("RESEARCH EXPERIMENT SUMMARY")
print("============================")

for name, filepath in files.items():

    path = Path(filepath)

    print(f"\n{name}")
    print("-" * len(name))

    if path.exists():
        df = pd.read_csv(path)
        print(df.to_string(index=False))
    else:
        print("Result file not found:", filepath)

# Export consolidated report
output_file = "results/master_research_summary.xlsx"

with pd.ExcelWriter(output_file) as writer:

    for sheet_name, filepath in files.items():

        path = Path(filepath)

        if path.exists():
            df = pd.read_csv(path)

            df.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False
            )

print("\nMaster research report saved:")
print(output_file)
