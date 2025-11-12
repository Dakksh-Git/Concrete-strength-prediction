import streamlit as st
import joblib
import pandas as pd

# Load trained model pipeline
model = joblib.load("concrete_rf_pipeline.joblib")

st.title("🏗️ Concrete Compressive Strength Predictor")
st.write("Enter the mix properties below:")

# Define input features
features = ["Cement","BlastFurnaceSlag","FlyAsh","Water",
            "Superplasticizer","CoarseAggregate","FineAggregate","Age"]

inputs = {}
for feature in features:
    inputs[feature] = st.number_input(feature, value=50.0)

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# Predict button
if st.button("Predict Strength"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Compressive Strength: {prediction:.2f} MPa")
# Bash streamlit run app.py in the terminal to run the website as local host or on external network.
