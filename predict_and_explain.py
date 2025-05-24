import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import shutil

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "xgboost_crop_dropout_trained.pkl")
db_path = os.path.join(base_dir, "Data/Crop_recommendation.csv")
bundle = joblib.load(model_path)
model = bundle["model"]
label_encoder = bundle["label_encoder"]
features = bundle["features"]

friendly_names = {
    "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
    "temperature": "Temperature", "humidity": "Humidity",
    "ph": "pH", "rainfall": "Rainfall"
}
modifiable_features = {"N", "P", "K", "ph"}

def generate_crop_recommendation(input_data, save_dir="results"):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for f in features:
        if f not in input_data:
            input_data[f] = np.nan

    X_input = pd.DataFrame([input_data])[features]
    X_input = X_input.astype(float)

    probs = model.predict_proba(X_input)[0]
    top_indices = probs.argsort()[-3:][::-1]

    explainer = shap.Explainer(model)
    shap_values = explainer(X_input)

    df = pd.read_csv(db_path)

    predictions = []
    report_lines = ["As per our prediction:\n"]

    for rank, idx in enumerate(top_indices):
        crop = label_encoder.inverse_transform([idx])[0]
        prob = round(probs[idx], 4)
        shap_vals = shap_values.values[0, :, idx]
        base_val = shap_values.base_values[0, idx]

        image_filename = f"{crop.lower().replace(' ', '_')}.png"
        image_path = os.path.join(save_dir, image_filename)
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=X_input.values[0],
            feature_names=X_input.columns
        )
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        top_pos = sorted([(f, shap_vals[i]) for i, f in enumerate(features) if shap_vals[i] > 0], key=lambda x: -x[1])[:2]
        top_neg = sorted([(f, shap_vals[i]) for i, f in enumerate(features) if shap_vals[i] < 0], key=lambda x: x[1])[:2]

        good_features = [friendly_names[f] for f, _ in top_pos]
        bad_features = [friendly_names[f] for f, _ in top_neg]

        good_text = " and ".join(good_features) if good_features else "unknown factors"
        bad_text = " and ".join(bad_features) if bad_features else "none"

        feature_means = df[df['label'] == crop][features].mean()
        suggestions = []
        for f in modifiable_features:
            if not pd.isna(input_data[f]):
                current = input_data[f]
                ideal = feature_means[f]
                direction = "increasing" if ideal > current else "decreasing"
                suggestions.append(f"{direction} {friendly_names[f]} to {ideal:.1f}(Currently : {input_data[f]})")

        crop_rank = f"{crop} is suggested with a probability of {prob * 100:.2f}%."
        reasons = f"The best factors supporting this are {good_text}."
        cautions = f"However, {bad_text} might hinder a good growth."
        recommendations = "For better results, consider :\n" + "\n".join(suggestions) + "."

        full_text = f"{crop_rank} {reasons} {cautions} {recommendations}"
        report_lines.append(full_text)

        predictions.append({
            "rank": rank + 1,
            "crop": crop,
            "probability": prob,
            "image_path": image_path,
            "report": full_text
        })

    missing_feats = [f for f in features if pd.isna(input_data[f])]
    if missing_feats:
        missing_names = ", ".join(friendly_names[f] for f in missing_feats)
        report_lines.append(f"\nNOTE: This prediction was made in absence of {missing_names}. For more accurate results, please rerun.")

    return {
        "top_predictions": predictions,
        "full_report": "\n\n".join(report_lines)
    }

if __name__ == "__main__":
    import sys
    import json
    import numpy as np

    user_input = json.loads(sys.argv[1])  
    result = generate_crop_recommendation(user_input)

    def clean_json(obj):
        if isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_json(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    print(json.dumps(clean_json(result)))
    sys.stdout.flush()
    sys.stderr.flush()
