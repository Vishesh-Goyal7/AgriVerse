import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

missing_per_label = 10

df = pd.read_csv("Data/Crop_recommendation.csv")

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

df_simulated = df.copy()

for label in df_simulated['label'].unique():
    label_indices = df_simulated[df_simulated['label'] == label].sample(n=missing_per_label, random_state=42).index
    df_simulated.loc[label_indices, ['humidity', 'rainfall']] = float('nan')

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df_simulated[features]
y = df_simulated['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric='mlogloss'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

bundle = {
    'model': model,
    'label_encoder': label_encoder,
    'features': features
}
joblib.dump(bundle, "xgboost_crop_dropout_trained.pkl")
