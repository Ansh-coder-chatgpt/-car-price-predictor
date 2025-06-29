import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ----------------- 🔰 Logo -----------------
try:
    logo = Image.open("turbocare_logo.jpg")  # Make sure this image is in same folder
    st.image(logo, width=200)
except FileNotFoundError:
    st.warning("⚠️ Logo not found. Please add 'turbocare_logo.jpg' to your folder.")

# ----------------- ⚙️ Page Setup -----------------
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Selling Price Predictor")
st.markdown("Enter the car details below to predict its estimated selling price.")

# ----------------- 📦 Load Model -----------------
model, feature_names = joblib.load('car_price_model.pkl')  # Trained model file

# ----------------- 🧾 User Inputs -----------------
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1, format="%.2f")
kms_driven = st.number_input("Kms Driven", min_value=0, step=100)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
age = st.number_input("Age of the Car (in years)", min_value=0, max_value=50)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# ----------------- 🔁 Feature Encoding -----------------
fuel_petrol = 1 if fuel_type == 'Petrol' else 0
fuel_diesel = 1 if fuel_type == 'Diesel' else 0
seller_individual = 1 if seller_type == 'Individual' else 0
trans_manual = 1 if transmission == 'Manual' else 0

# Final feature vector
input_data = [present_price, kms_driven, owner, age,
              fuel_diesel, fuel_petrol, seller_individual, trans_manual]
input_df = pd.DataFrame([input_data], columns=feature_names)

# ----------------- 🔮 Prediction Button -----------------
if st.button("Predict Price"):

    # ✅ Strong Validation
    if present_price < 1:
        st.error("❌ Present Price must be at least ₹1 lakh.")
    elif kms_driven < 1000:
        st.error("❌ Kms Driven should be realistic (1000+).")
    elif age < 0:
        st.error("❌ Car age cannot be negative.")
    else:
        # ✅ Run prediction
        prediction = model.predict(input_df)[0]
        user_pred = round(prediction, 2)
        st.success(f"💰 Estimated Selling Price: ₹ {user_pred:.2f} lakhs")

        # ----------------- 📊 Price Range Comparison -----------------
        st.subheader("📊 Price Range Comparison")

        min_price = 1
        avg_price = 5
        max_price = 12

        categories = ['Min Price', 'Avg Price', 'Max Price', 'Your Car']
        values = [min_price, avg_price, max_price, user_pred]
        colors = ["#FC0E0E", "#D0F84E", "#23C265", "#4034E0"]

        fig, ax = plt.subplots()
        bars = ax.bar(categories, values, color=colors)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + 0.1, yval + 0.2, f'₹{yval}L', fontsize=10)

        ax.set_ylabel('Selling Price (Lakhs)')
        ax.set_ylim([0, max(values) + 2])
        st.pyplot(fig)
