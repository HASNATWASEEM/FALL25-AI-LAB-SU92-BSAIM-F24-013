# Diabetes Prediction Flask App (Auto-generated)

This project was generated automatically based on your uploaded dataset located at `/mnt/data/diabetes_dataset.csv`.

## Files created under /mnt/data/diabetes_project
- `model/` : Contains `model.pkl`, `scaler.pkl`, and `metadata.json`
- `templates/index.html` : Web form for inputs and prediction display
- `app.py` : Flask application to serve predictions
- `requirements.txt` : Python dependencies
- `data_processed.csv` : Processed dataset used for training
- `diabetes_project.zip` : Zipped project for download (in /mnt/data/diabetes_project.zip)

## How to run locally
1. Create virtual environment & activate it.
2. `pip install -r requirements.txt`
3. Run `python app.py`
4. Open `http://127.0.0.1:5000/` in your browser.

Note: The original feature columns used are:
['Age', 'Sex', 'Ethnicity', 'BMI', 'Waist_Circumference', 'Fasting_Blood_Glucose', 'HbA1c', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 'Cholesterol_Total', 'Cholesterol_HDL', 'Cholesterol_LDL', 'GGT', 'Serum_Urate', 'Physical_Activity_Level', 'Dietary_Intake_Calories', 'Alcohol_Consumption', 'Smoking_Status', 'Family_History_of_Diabetes']

Metadata is in `model/metadata.json`.
