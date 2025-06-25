import streamlit as st
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📱 Telco Churn Predictor (5 Features)")

# Collect inputs for 5 features
tenure = st.number_input("Tenure (months)", min_value=0.0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# You can use selectbox for categorical features
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Convert categories to numeric values (must match training encoding)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

# Prepare data
input_data = [
    tenure,
    monthly_charges,
    total_charges,
    contract_map[contract],
    internet_map[internet]
]

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict([input_data])[0]
    result = "Will Churn" if prediction == 1 else "Will Not Churn"
    st.success(f"📊 Prediction: {result}")
