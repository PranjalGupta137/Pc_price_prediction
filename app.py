import streamlit as st
import joblib
import numpy as np

# 1. Load Model aur Encoders
model = joblib.load("Pc_price_prediction.pkl")
encoder_cpu = joblib.load("cpu_encoder.pkl")
encoder_gpu = joblib.load("gpu_encoder.pkl")

st.title("💻 Advanced Laptop Price Predictor")

# 2. Inputs
ram = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5)

# CPU aur GPU options (Encoders se list nikal rahe hain)
cpu = st.selectbox("Select CPU", encoder_cpu.classes_)
gpu = st.selectbox("Select GPU Brand", encoder_gpu.classes_)

if st.button("Predict Price"):
    # Text ko wapis numbers mein badalna
    cpu_encoded = encoder_cpu.transform([cpu])[0]
    gpu_encoded = encoder_gpu.transform([gpu])[0]
    
    # Prediction
    input_data = np.array([[ram, weight, cpu_encoded, gpu_encoded]])
    prediction = model.predict(input_data)
    
    st.success(f"Estimated Price: ₹{int(prediction[0])}")