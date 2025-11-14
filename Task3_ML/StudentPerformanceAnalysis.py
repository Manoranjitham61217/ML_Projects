import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
selector = joblib.load(os.path.join(BASE_DIR, "feature_selector.pkl"))
x_columns = joblib.load(os.path.join(BASE_DIR, "x_column.pkl"))

st.sidebar.header("Input Features")

input_values = {}

for col in x_columns:
    col_name = str(col)
    input_values[col_name] = st.sidebar.number_input(
        col_name,
        value=0.0,
        format="%.2f"
    )



st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("🎓 Student Performance Prediction Web App")
st.write("Enter the student details below to predict performance.")


input_df = pd.DataFrame([input_values])
scaled_data = scaler.transform(input_df)

selected_data = selector.transform(scaled_data)

if st.button("Predict"):
    prediction = model.predict(selected_data)[0]
    st.success(f"Predicted Performance: {prediction:.2f}")
    if(prediction>=95):
        feedback="Your Performance was very Excellent"
    elif(prediction>=75 and prediction<95):
        feedback="Your Performance was Good"
    elif(prediction>=50 and prediction<75):
        feedback="Your Preformance was Average need some Improvement"
    else:
        feedback="Got Fail need to make Practice"

    st.success(feedback)