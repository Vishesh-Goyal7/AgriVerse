import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("Data/Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

model = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric="mlogloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

bundle = {
    "model": model,
    "label_encoder": label_encoder,
    "features": list(X.columns)
}

joblib.dump(bundle, "xgboost_crop_bundle.pkl")
