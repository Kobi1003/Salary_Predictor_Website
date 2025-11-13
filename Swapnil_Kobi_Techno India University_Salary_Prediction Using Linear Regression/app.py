import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessor
model = joblib.load("salary_predictor.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("💰 Salary Prediction App")

# Create input fields for user
st.subheader("Enter Employee Details")

years_exp = st.number_input("Years of Experience", min_value=0.0, step=0.1)
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
job = st.selectbox("Job Title", ["Analyst", "Specialist", "Manager", "Director", "Intern"])
dept = st.selectbox("Department", ["Engineering", "Finance", "Sales", "Marketing", "HR"])

# When user clicks "Predict"
if st.button("Predict Salary"):
    input_df = pd.DataFrame({
        "YearsExperience": [years_exp],
        "EducationLevel": [education],
        "JobTitle": [job],
        "Department": [dept]
    })

    # Preprocess input
    input_encoded = pd.get_dummies(input_df)
    
    # Align columns with model training data (fill missing columns with 0)
    model_columns = joblib.load("model_columns.pkl")  # Save these during training
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    predicted_salary = model.predict(input_encoded)[0]
    st.success(f"💰 Estimated Salary: ${predicted_salary:,.2f}")
