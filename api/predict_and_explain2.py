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

def generate_global_feature_importance(save_path="results/global_importance.png"):
    df = pd.read_csv("Data/Crop_recommendation.csv")
    X = df[features]

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    if len(shap_values.shape) == 3:
        mean_importance = np.mean(np.abs(shap_values.values), axis=(0, 2))
    else:
        mean_importance = np.mean(np.abs(shap_values.values), axis=0)

    sorted_indices = np.argsort(mean_importance)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importance = mean_importance[sorted_indices]

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_features[::-1], sorted_importance[::-1], color="#6BA368")
    plt.xlabel("Mean |SHAP value| (avg impact on model output)")
    plt.xticks([])
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def calculate_crop_ideals(csv_path="Data/Crop_recommendation.csv"):
    df = pd.read_csv(csv_path)
    return df.groupby("label")[features].mean().to_dict(orient="index")

def generate_crop_recommendation(input_data, save_dir="results"):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    generate_global_feature_importance()

    for f in features:
        if f not in input_data:
            input_data[f] = np.nan

    X_input = pd.DataFrame([input_data])[features]
    X_input = X_input.astype(float)

    probs = model.predict_proba(X_input)[0]
    top_indices = probs.argsort()[-3:][::-1]

    # Calculate trust score
    prob_top_1 = probs[top_indices[0]]
    prob_top_2 = probs[top_indices[1]]
    confidence_margin = round(float(prob_top_1 - prob_top_2), 4)

    if confidence_margin >= 0.5:
        trust_level = "High"
    elif confidence_margin >= 0.25:
        trust_level = "Medium"
    else:
        trust_level = "Low"

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

        feature_impact = []
        for i, f in enumerate(features):
            impact = {
                "feature": friendly_names[f],
                "value": input_data[f] if not pd.isna(input_data[f]) else None,
                "shap": round(float(shap_vals[i]), 5)
            }
            feature_impact.append(impact)

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
            "report": full_text,
            "feature_impact": feature_impact
        })

    missing_feats = [f for f in features if pd.isna(input_data[f])]
    if missing_feats:
        missing_names = ", ".join(friendly_names[f] for f in missing_feats)
        report_lines.append(f"\nNOTE: This prediction was made in absence of {missing_names}. For more accurate results, please rerun.")

    
    # Calculate counterfactual suggestion
    crop_ideals = calculate_crop_ideals()
    top_crop = label_encoder.inverse_transform([0])[0]

    min_deviation = float('inf')
    best_crop = None
    best_changes = []

    for crop, ideal_vals in crop_ideals.items():
        if crop == top_crop:
            continue  # skip the already predicted crop

        total_deviation = 0
        change_list = []

        for f in modifiable_features:
            input_val = input_data[f]
            ideal_val = ideal_vals[f]

            if pd.isna(input_val) or pd.isna(ideal_val) or ideal_val == 0:
                continue

            percent_dev = abs((input_val - ideal_val) / ideal_val) * 100
            total_deviation += percent_dev

            change_list.append({
                "feature": friendly_names[f],
                "current": round(float(input_val), 2),
                "ideal": round(float(ideal_val), 2),
                "change": round(float(ideal_val - input_val), 2)
            })

        if total_deviation < min_deviation:
            min_deviation = total_deviation
            best_crop = crop
            best_changes = change_list

    counterfactual_suggestion = {
        "alternative_crop": best_crop,
        "percent_deviation": round(min_deviation, 2),
        "suggested_changes": best_changes
    }
    
    return {
        "top_predictions": predictions,
        "full_report": "\n\n".join(report_lines),
        "global_importance_path": "results/global_importance.png",
        "trust_score": {
            "level": trust_level,
            "confidence": confidence_margin,
            "counterfactual_suggestion": counterfactual_suggestion
        }
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