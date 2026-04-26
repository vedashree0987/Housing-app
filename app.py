import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("house_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 House Price Prediction")

st.write("Enter house details:")

# Inputs (based on YOUR dataset)
sqft = st.number_input("Square Footage")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
year = st.number_input("Year Built")
lot = st.number_input("Lot Size")
garage = st.number_input("Garage Size")
quality = st.number_input("Neighborhood Quality")

if st.button("Predict Price"):

    data = np.array([[sqft, bedrooms, bathrooms, year, lot, garage, quality]])
    
    # Scale
    data_scaled = scaler.transform(data)
    
    # Predict
    prediction = model.predict(data_scaled)
    
    # Reverse log transformation
    price = np.exp(prediction[0])

    st.success(f"💰 Predicted House Price: {price:,.2f}")
