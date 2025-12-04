from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load pre-trained model
model = joblib.load("diabetes_model.pkl")

# Load dataset for dynamic dropdowns
df = pd.read_csv("data.csv")

# Columns for dropdowns (categorical)
exclude_cols = ['Age','Sex','Ethnicity','BMI','Waist_Circumference','Fasting_Blood_Glucose',
                'HbA1c','Blood_Pressure_Systolic','Blood_Pressure_Diastolic',
                'Cholesterol_Total','Cholesterol_HDL','Cholesterol_LDL','GGT','Serum_Urate',
                'Physical_Activity_Level','Dietary_Intake_Calories','Alcohol_Consumption',
                'Smoking_Status','Family_History_of_Diabetes','Previous_Gestational_Diabetes',
                'Diabetes_Risk']

dropdown_features = [col for col in df.columns if col not in exclude_cols]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    # Prepare dropdown options
    dropdown_options = {}
    for feature in dropdown_features:
        dropdown_options[feature] = sorted(df[feature].dropna().unique())

    if request.method == "POST":
        # Gather input values
        input_data = {}
        for col in df.columns[:-1]:
            value = request.form.get(col)
            # Convert numeric columns to float
            try:
                input_data[col] = float(value)
            except:
                input_data[col] = value

        # Convert categorical to numeric using factorize (matching training)
        input_df = pd.DataFrame([input_data])
        cat_cols = input_df.select_dtypes(include='object').columns
        for col in cat_cols:
            input_df[col] = pd.factorize(input_df[col])[0]

        # Predict
        prediction = model.predict(input_df)[0]

    return render_template("index.html", dropdown_options=dropdown_options, prediction=prediction, columns=df.columns[:-1])

if __name__ == "__main__":
    app.run(debug=True)
