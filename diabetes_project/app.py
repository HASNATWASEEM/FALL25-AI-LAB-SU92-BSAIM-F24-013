from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

# features expected by the model after encoding
ENCODED_FEATURES = meta["encoded_features"]
ORIG_FEATURES = meta["original_features"]
CATEGORICALS = meta["categorical_columns"]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form values using original feature names
        data = {}
        for f in ORIG_FEATURES:
            v = request.form.get(f)
            if v is None or v == "":
                return render_template("index.html", error=f"Missing value for {f}")
            data[f] = v

        # Build DataFrame
        df = pd.DataFrame([data])
        # Convert numeric columns to float where possible
        for c in df.columns:
            try:
                df[c] = df[c].astype(float)
            except Exception:
                pass

        # One-hot encode categorical columns like during training
        df_encoded = pd.get_dummies(df, columns=CATEGORICALS, drop_first=True)

        # Ensure all encoded columns present and aligned in same order
        for col in ENCODED_FEATURES:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[ENCODED_FEATURES]

        # Scale and predict
        X_scaled = scaler.transform(df_encoded)
        proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None
        pred = int(model.predict(X_scaled)[0])
        label = "Diabetic" if pred == 1 else "Non-Diabetic"
        prob_pct = round(proba*100,2) if proba is not None else None
        return render_template("index.html", prediction=True, label=label, prob=prob_pct, inputs=data)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
