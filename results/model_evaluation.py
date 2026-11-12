import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load dataset
data = pd.read_csv("data/spectrum_data.csv")

X = data[["snr", "noise_power", "interference", "energy"]]
y = data["occupancy"]

# Same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained model
model = joblib.load("models/spectrum_model.pkl")

# Predictions
predictions = model.predict(X_test)

# Print metrics
print("MODEL PERFORMANCE EVALUATION")
print("-----------------------------")
print(classification_report(y_test, predictions))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Available", "Occupied"]
)

display.plot()
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig("figures/confusion_matrix.png", dpi=300)
plt.show()

print("Confusion matrix saved successfully!")