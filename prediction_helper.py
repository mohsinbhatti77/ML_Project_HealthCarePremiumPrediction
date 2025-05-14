# codebasics ML course: codebasics.io, all rights reserverd

import pandas as pd
import joblib

model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")
scaler_young = joblib.load("artifacts/scaler_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score


def preprocess_input(input_dict):
    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    # Replace None with default values
    age = input_dict.get('Age', 25)  # Default age 25
    number_of_dependants = input_dict.get('Number of Dependants', 0)
    income_lakhs = input_dict.get('Income in Lakhs', 10)
    genetical_risk = input_dict.get('Genetical Risk', 2)
    insurance_plan = insurance_plan_encoding.get(input_dict.get('Insurance Plan', 'Bronze'), 1)
    medical_history = input_dict.get('Medical History', 'No Disease') or 'No Disease'

    df = pd.DataFrame(0, columns=[
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk',
        'total_risk_score', 'normalized_risk_score', 'gender_Male', 'region_Northwest',
        'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional', 'smoking_status_Regular', 'employment_status_Salaried',
        'employment_status_Self-Employed'
    ], index=[0])

    df['age'] = age
    df['number_of_dependants'] = number_of_dependants
    df['income_lakhs'] = income_lakhs
    df['genetical_risk'] = genetical_risk
    df['insurance_plan'] = insurance_plan

    # Compute total and normalized risk scores
    risk_scores = {"diabetes": 6, "heart disease": 8, "high blood pressure": 6, "thyroid": 5, "no disease": 0}
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    df['total_risk_score'] = total_risk_score
    df['normalized_risk_score'] = total_risk_score / 14

    # One-hot encoding
    categorical_mappings = {
        'Gender': {'Male': 'gender_Male'},
        'Region': {'Northwest': 'region_Northwest', 'Southeast': 'region_Southeast', 'Southwest': 'region_Southwest'},
        'Marital Status': {'Unmarried': 'marital_status_Unmarried'},
        'BMI Category': {'Obesity': 'bmi_category_Obesity', 'Overweight': 'bmi_category_Overweight',
                         'Underweight': 'bmi_category_Underweight'},
        'Smoking Status': {'Occasional': 'smoking_status_Occasional', 'Regular': 'smoking_status_Regular'},
        'Employment Status': {'Salaried': 'employment_status_Salaried',
                              'Self-Employed': 'employment_status_Self-Employed'}
    }

    for key, mapping in categorical_mappings.items():
        if key in input_dict and input_dict[key] in mapping:
            df[mapping[input_dict[key]]] = 1

    df = handle_scaling(age, df)

    return df

def handle_scaling(age, df):
    # scale age and income_lakhs column
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None # since scaler object expects income_level supply it. This will have no impact on anything
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])