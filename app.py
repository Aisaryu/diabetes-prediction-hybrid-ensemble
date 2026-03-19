import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the saved artifacts
@st.cache_resource
def load_models():
    model = joblib.load('hybrid_ensemble_model.pkl')
    scaler = joblib.load('scaler.pkl')
    imputer = joblib.load('imputer.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, imputer, features

model, scaler, imputer, feature_names = load_models()

# 2. App UI Setup
st.title("🩺 Type 2 Diabetes Prediction Prototype")
st.markdown("Powered by **Hybrid Gaussian-Adaptive SMOTE & Ensemble Learning**")
st.divider()

st.sidebar.header("Patient Health Indicators")
st.sidebar.write("Adjust the parameters to see the real-time prediction.")

# 3. Create interactive inputs for the user
# Note: The CDC dataset has 21 features. For a clean UI, we highlight the most critical ones 
# and set the rest to default medians, OR you can generate sliders for all of them.
user_data = {}

# Example of key interactive features (modify based on your feature names)
user_data['HighBP'] = st.sidebar.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
user_data['HighChol'] = st.sidebar.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
user_data['BMI'] = st.sidebar.slider("BMI", 10.0, 98.0, 25.0)
user_data['Smoker'] = st.sidebar.selectbox("Smoker (100+ cigarettes in life)", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
user_data['Age'] = st.sidebar.slider("Age Category (1-13)", 1, 13, 5) # CDC uses 13-level age category
user_data['GenHlth'] = st.sidebar.slider("General Health (1: Excellent -> 5: Poor)", 1, 5, 2)

# Fill remaining features with default/median values to satisfy the model's 21-feature requirement
for feature in feature_names:
    if feature not in user_data:
        user_data[feature] = 0 # Replace 0 with the actual median of your dataset for better accuracy

# 4. Predict Button
if st.button("Predict Diabetes Risk", type="primary"):
    # Convert input to dataframe in the exact order of training features
    input_df = pd.DataFrame([user_data], columns=feature_names)
    
    # Apply Pipeline (Impute -> Scale)
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    
    # Get Probability and Prediction
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Use your optimized threshold from your research (e.g., 0.35 instead of default 0.5)
    OPTIMIZED_THRESHOLD = 0.35 
    prediction = 1 if probability >= OPTIMIZED_THRESHOLD else 0

    # 5. Display Results
    st.subheader("Results:")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("🚨 **High Risk of Type 2 Diabetes**")
        else:
            st.success("✅ **Low Risk of Type 2 Diabetes**")
            
    with col2:
        st.metric(label="Probability Score", value=f"{probability * 100:.1f}%")
        st.progress(float(probability))
        
    st.info(f"*(Note: Model threshold is set to {OPTIMIZED_THRESHOLD} based on thesis optimization)*")