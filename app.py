
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
bundle = joblib.load("bankruptcy_xgboost.pkl")
model = bundle["model"]
features = bundle["features"]
class_names = bundle["class_names"]

# Page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Bankruptcy Prevention System")
st.write(
    "Enter the company's risk indicators to predict "
    "whether it is likely to face bankruptcy."
)

st.info("Enter a value from 0 to 1 for each risk indicator.")

# Input form
with st.form("prediction_form"):
    industrial_risk = st.selectbox(
        "Industrial Risk", [0.0, 0.5, 1.0]
    )
    management_risk = st.selectbox(
        "Management Risk", [0.0, 0.5, 1.0]
    )
    financial_flexibility = st.selectbox(
        "Financial Flexibility", [0.0, 0.5, 1.0]
    )
    credibility = st.selectbox(
        "Credibility", [0.0, 0.5, 1.0]
    )
    competitiveness = st.selectbox(
        "Competitiveness", [0.0, 0.5, 1.0]
    )
    operating_risk = st.selectbox(
        "Operating Risk", [0.0, 0.5, 1.0]
    )

    submitted = st.form_submit_button("Predict Bankruptcy")

if submitted:
    input_values = {
        "industrial_risk": industrial_risk,
        "management_risk": management_risk,
        "financial_flexibility": financial_flexibility,
        "credibility": credibility,
        "competitiveness": competitiveness,
        "operating_risk": operating_risk
    }

    input_data = pd.DataFrame(
        [[input_values[feature] for feature in features]],
        columns=features
    )

    prediction = int(model.predict(input_data)[0])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Bankruptcy")
    else:
        st.success("Prediction: Non-bankruptcy")

    st.caption(
        "This is a machine-learning prediction, not a guarantee "
        "of the company's financial outcome."
    )
