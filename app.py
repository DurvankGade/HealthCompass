import streamlit as st
import pandas as pd
import joblib
import json

# Let's set up the basic look of the page.
st.set_page_config(page_title="Patient Risk Predictor", layout="wide")

# Load the trained model and feature list.
# Cache the model so it only loads once.
@st.cache_resource
def load_model():
    """Loads the model and column list."""
    model = joblib.load('chronic_patient_risk_model.joblib')
    with open('model_columns.json', 'r') as f:
        columns = json.load(f)
    return model, columns

model, model_columns = load_model()

# Time for the main title and a quick intro for the user.
st.title("HealthCompass 🩺: Chronic Patient Risk Prediction Engine")
st.markdown("Use the sliders on the left to enter patient data. We'll predict their 90-day deterioration risk.")

# --- All the user inputs will live in the sidebar ---
st.sidebar.header("Patient Input Features")

def get_user_inputs():
    """This function just creates all the interactive widgets in the sidebar."""
    inputs = {}
    
    # Sliders and dropdowns for the basic patient info.
    inputs['age'] = st.sidebar.slider("Age", 40, 100, 65)
    inputs['gender'] = 1 if st.sidebar.selectbox("Gender", ["Female", "Male"]) == "Male" else 0
    inputs['smoker'] = 1 if st.sidebar.selectbox("Smoker", ["No", "Yes"]) == "Yes" else 0
    inputs['bmi'] = st.sidebar.slider("BMI", 15.0, 50.0, 28.5)
    
    # A new section for their recent health stats.
    st.sidebar.subheader("Recent 30-Day Averages & Stability")
    inputs['avg_hr'] = st.sidebar.slider("Average Heart Rate (bpm)", 50, 120, 75)
    inputs['std_hr'] = st.sidebar.slider("Heart Rate Stability (Std Dev)", 0.0, 15.0, 3.5)
    inputs['avg_bp_sys'] = st.sidebar.slider("Average Systolic BP (mmHg)", 90, 180, 135)
    inputs['std_bp_sys'] = st.sidebar.slider("Systolic BP Stability (Std Dev)", 0.0, 25.0, 8.0)
    inputs['avg_glucose'] = st.sidebar.slider("Average Glucose (mg/dL)", 70, 250, 110)
    inputs['avg_spo2'] = st.sidebar.slider("Average SpO2 (%)", 90.0, 100.0, 97.5)
    
    # And another section for their health trends.
    st.sidebar.subheader("Recent 30-Day Trends (Slope)")
    inputs['slope_hr'] = st.sidebar.slider("Heart Rate Trend", -2.0, 2.0, 0.1)
    inputs['slope_bp_sys'] = st.sidebar.slider("Systolic BP Trend", -3.0, 3.0, 0.5)
    inputs['slope_glucose'] = st.sidebar.slider("Glucose Trend", -5.0, 5.0, 0.2)
    
    st.sidebar.subheader("Lifestyle")
    inputs['adherence_rate'] = st.sidebar.slider("Medication Adherence Rate", 0.0, 1.0, 0.85)
    inputs['avg_exercise'] = st.sidebar.slider("Average Daily Exercise (mins)", 0, 120, 25)
    
    return inputs

# Grab all the values from the sidebar.
user_inputs = get_user_inputs()

# Make a big, clickable button to kick off the prediction.
predict_button = st.button("Predict Risk", type="primary")

# Only run the prediction logic if the button has been clicked.
if predict_button:
    # We need to wrap the user's inputs in a DataFrame for the model.
    input_df = pd.DataFrame([user_inputs])
    # This next line is CRUCIAL - it makes sure the columns are in the exact same
    # order the model was trained on. Otherwise, you get chaos.
    input_df = input_df[model_columns]

    # Get the model's prediction (it gives a probability).
    risk_proba = model.predict_proba(input_df)[0, 1]
    risk_score = int(risk_proba * 100)

    # Based on the score, decide what to show the user.
    if risk_score >= 75:
        risk_level = "High Risk"
        color = "red"
        recommendation = "Urgent: Review patient's file and consider scheduling an immediate consultation."
    elif risk_score >= 40:
        risk_level = "Medium Risk"
        color = "orange"
        recommendation = "Action: Schedule a follow-up call within the next week to discuss vitals and adherence."
    else:
        risk_level = "Low Risk"
        color = "green"
        recommendation = "Action: Continue with standard monitoring and care plan."
        
    # Show the final score in a nice big "metric" card.
    st.metric(label="Predicted Deterioration Risk", value=f"{risk_score}%", delta=risk_level)
    
    # Use some HTML to show the risk level in a matching color.
    st.markdown(f"## <span style='color:{color};'>{risk_level}</span>", unsafe_allow_html=True)
    st.info(f"**Recommended Action:** {recommendation}")

    # Display the key risk drivers directly.
    st.subheader("Key Risk Drivers")
    st.write("Here are some of the factors that might be influencing the prediction:")
    
    # This isn't SHAP, just some simple if-statements to provide basic insights.
    if user_inputs['adherence_rate'] < 0.7:
        st.warning("🔴 **Poor Medication Adherence** is a major contributor to the risk.")
    if user_inputs['slope_bp_sys'] > 0.4:
        st.warning("🔴 **Rising Systolic Blood Pressure Trend** is increasing the risk.")
    if user_inputs['avg_glucose'] > 150:
        st.warning("🟡 **High Average Glucose** is a contributing factor.")
    if user_inputs['avg_exercise'] > 30 and user_inputs['adherence_rate'] > 0.9:
        st.success("🟢 **Good Adherence and Exercise** are helping to lower the risk.")

