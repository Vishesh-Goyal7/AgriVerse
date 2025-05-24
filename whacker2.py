import pandas as pd
import joblib
import random
from collections import Counter

bundle = joblib.load("xgboost_crop_dropout_trained.pkl")
model = bundle["model"]
label_encoder = bundle["label_encoder"]
all_features = bundle["features"]

df = pd.read_csv("Data/Crop_recommendation.csv")
num_samples = 1000
samples = df.sample(n=num_samples, random_state=1)

correct = 0
missing_feature_counter = Counter()
skewed_feature_counter = Counter()

for idx, row in samples.iterrows():
    input_features = row[all_features].copy()
    true_label = row["label"]

    removed_features = random.sample(all_features, 2)
    for feat in removed_features:
        missing_feature_counter[feat] += 1
        input_features[feat] = float('nan')

    X_input = pd.DataFrame([input_features])[all_features]

    pred_encoded = model.predict(X_input)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    if pred_label == true_label:
        correct += 1
    else:
        for feat in removed_features:
            skewed_feature_counter[feat] += 1

    print(f"True: {true_label:15} | Predicted: {pred_label:15} | Missing: {removed_features}")

print(f"\nAccuracy on {num_samples} samples with 2 missing features each: {correct}/{num_samples} = {correct/num_samples:.2%}")

print("\n🔍 Feature Sensitivity Analysis (Skew Rate per Missing Feature):")
for feature in all_features:
    total_missing = missing_feature_counter[feature]
    total_skewed = skewed_feature_counter[feature]
    if total_missing > 0:
        risk_ratio = (total_skewed / total_missing) * 100
        print(f"Missing Feature: {feature:15} | Skewed: {total_skewed:2d} / {total_missing:2d} | Risk: {risk_ratio:.2f}%")
