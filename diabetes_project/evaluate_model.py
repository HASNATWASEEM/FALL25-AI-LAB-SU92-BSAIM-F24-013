import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# ===============================
#   PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ===============================
#   LOAD MODEL, SCALER, META
# ===============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

ENCODED_FEATURES = meta["encoded_features"]
ORIG_FEATURES = meta["original_features"]
CATEGORICALS = meta["categorical_columns"]
TARGET = meta["target_column"]  # Make sure your metadata contains the target column

# ===============================
#   LOAD DATA
# ===============================
# Replace 'dataset.csv' with your dataset file
df = pd.read_csv("dataset.csv")

# Check if target column exists
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset!")

# Separate features and target
X = df[ORIG_FEATURES]
y = df[TARGET]

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=CATEGORICALS, drop_first=True)

# Ensure all encoded columns exist
for col in ENCODED_FEATURES:
    if col not in X_encoded.columns:
        X_encoded[col] = 0
X_encoded = X_encoded[ENCODED_FEATURES]

# Scale
X_scaled = scaler.transform(X_encoded)

# Predict
y_pred = model.predict(X_scaled)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")
