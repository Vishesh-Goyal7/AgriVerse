import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

bundle = joblib.load("xgboost_crop_dropout_trained.pkl")
model = bundle["model"]
label_encoder = bundle["label_encoder"]
all_features = bundle["features"]

df = pd.read_csv("Data/Crop_recommendation.csv")
num_samples = 1000
df = df.sample(n=num_samples, random_state=5)
X = df[all_features]
y_true = df["label"]
y_true_encoded = label_encoder.transform(y_true)

y_pred_encoded = model.predict(X)

y_pred = label_encoder.inverse_transform(y_pred_encoded)

print("🔍 Overall Evaluation Metrics:\n")
print(f"✅ Accuracy: {accuracy_score(y_true_encoded, y_pred_encoded):.4f}\n")
print(classification_report(y_true_encoded, y_pred_encoded, target_names=label_encoder.classes_))

cm = confusion_matrix(y_true_encoded, y_pred_encoded)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("📊 Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()