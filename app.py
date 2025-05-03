import streamlit as st
import pandas as pd
import joblib

# Load the saved model
clf = joblib.load('loan_model9.pkl')

st.title("Loan Eligibility Checker")
st.write("Enter applicant details to predict approval probability and status.")

# Sidebar inputs
st.sidebar.header("Applicant Info")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
app_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapp_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amt_rupees = st.sidebar.number_input("Loan Amount (₹)", min_value=0,value=120)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=0)
credit_hist = st.sidebar.selectbox("Credit History", [0.0, 1.0])
prop_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Build input DataFrame
input_df = pd.DataFrame([{
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_emp,
    'ApplicantIncome': app_income,
    'CoapplicantIncome': coapp_income,
    'LoanAmount': loan_amt_rupees,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_hist,
    'Property_Area': prop_area
}])

if st.button("Predict"):
    # Get prediction and probability
    pred = clf.predict(input_df)[0]
    proba = clf.predict_proba(input_df)[0]

    # Format results
    status = "Approved (Y)" if pred == 1 else "Rejected (N)"
    prob_approval = proba[1] * 100
    prob_rejection = proba[0] * 100

    st.markdown(f"## Result: {status}")
    st.markdown(f"- Approval probability: **{prob_approval:.2f}%**")
    st.markdown(f"- Rejection probability: **{prob_rejection:.2f}%**")
