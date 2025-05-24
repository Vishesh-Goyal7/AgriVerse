import joblib
import pandas as pd
import shap
import numpy as np

bundle = joblib.load("xgboost_crop_dropout_trained.pkl")
model = bundle["model"]
label_encoder = bundle["label_encoder"]
features = bundle["features"]

input_data = {
    'N': 66,
    'P': 52,
    'K': 49,
    'temperature': 22.0,
    'humidity': 85.0,
    'ph': 6.5,
    'rainfall': 200
}
X_input = pd.DataFrame([input_data])[features]

probs = model.predict_proba(X_input)[0]
top_indices = probs.argsort()[-3:][::-1]

explainer = shap.Explainer(model)
shap_values = explainer(X_input)

df = pd.read_csv("Data/Crop_recommendation.csv")
feature_means = df[features].mean()

friendly_names = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "pH",
    "rainfall": "Rainfall"
}

modifiable_features = {"N", "P", "K", "ph"}

report = []
for rank, idx in enumerate(top_indices):
    crop = label_encoder.inverse_transform([idx])[0]
    prob = probs[idx]
    shap_vals = shap_values.values[0, :, idx]
    
    top_pos = sorted(zip(features, shap_vals), key=lambda x: -x[1])[:2]
    top_neg = sorted(zip(features, shap_vals), key=lambda x: x[1])[:2]
    
    rank_label = ["most", "second most", "third most"][rank]
    section = f"The {rank_label} recommended crop with a probability of {prob:.2f} is **{crop}**."
    
    if top_pos:
        reasons = [f"{friendly_names[f]} being {input_data[f]}" for f, _ in top_pos]
        section += f" This is primarily due to " + ", and ".join(reasons) + "."

    suggestions = []
    for f, _ in top_neg:
        if f in modifiable_features:
            current = input_data[f]
            ideal = feature_means[f]
            diff = "higher" if ideal > current else "lower"
            suggestions.append(f"adjusting {friendly_names[f]} towards {ideal:.1f} ({diff} than current)")
    
    if suggestions:
        section += f" For better results, consider " + ", and ".join(suggestions) + "."

    report.append(section)

full_report = "\n\n".join(report)
print("\n📄 AI Crop Recommendation Report:\n")
print(full_report)
