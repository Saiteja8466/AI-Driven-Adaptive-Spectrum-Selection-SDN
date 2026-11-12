
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# Load dataset
data = pd.read_csv("data/interference_dataset.csv")

X = data[
    [
        "snr_db",
        "noise_power",
        "traffic_load",
        "num_interferers",
        "received_power"
    ]
]

y = data["interference_level"]

# Same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained model
model = joblib.load(
    "models/interference_prediction_model.pkl"
)

# Generate predictions
y_pred = model.predict(X_test)

# Print report
print("INTERFERENCE MODEL EVALUATION")
print("--------------------------------")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Low", "Medium", "High"]
))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Low", "Medium", "High"],
    cmap="Blues"
)

plt.title("AI-Based Interference Classification")
plt.tight_layout()

plt.savefig("figures/interference_confusion_matrix.png", dpi=300)

plt.show()

print("Confusion matrix saved successfully!")
