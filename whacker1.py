### Tested Successfully. Not needed anymore

import joblib
import pandas as pd
import numpy as np

bundle = joblib.load("xgboost_crop_bundle.pkl")
model = bundle["model"]
label_encoder = bundle["label_encoder"]
features = bundle["features"]

input_data = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 22.0,
    'humidity': 80.0,
    'ph': 6.5,
    'rainfall': 200.0
}

X_input = pd.DataFrame([input_data])[features]

pred_encoded = model.predict(X_input)[0]
pred_crop = label_encoder.inverse_transform([pred_encoded])[0]

print(f"Recommended Crop: {pred_crop}")
