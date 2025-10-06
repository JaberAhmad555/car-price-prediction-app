
import streamlit as st
import pandas as pd
import shap
import joblib

# Load the trained model and the SHAP explainer
model = joblib.load("car_price_model(1).pkl")
explainer = joblib.load("shap_explainer.pkl")

st.title("Car Price Prediction App")

# --- User Input Sidebar ---
st.sidebar.header("Enter Car Details")
year = st.sidebar.number_input("Year", 2000, 2025, 2018)
present_price = st.sidebar.number_input("Present Price (in Lakhs)", 0.0, 50.0, 5.0, step=0.1)
mileage = st.sidebar.number_input("Mileage (km)", 0, 300000, 50000)
fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Number of Owners", [0, 1, 2, 3])

# --- Preprocessing Input ---
input_data = pd.DataFrame([{
    "Year": year,
    "Present_Price": present_price,
    "Kms_Driven": mileage,
    "Fuel_Type": fuel,
    "Seller_Type": seller_type,
    "Transmission": transmission,
    "Owner": owner
}])

fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_map = {'Dealer': 0, 'Individual': 1}
transmission_map = {'Manual': 0, 'Automatic': 1}

input_data['Fuel_Type'] = input_data['Fuel_Type'].map(fuel_map)
input_data['Seller_Type'] = input_data['Seller_Type'].map(seller_map)
input_data['Transmission'] = input_data['Transmission'].map(transmission_map)

# --- Prediction ---
try:
    pred_price = model.predict(input_data)[0]
    st.subheader(f"Predicted Price: ₹{pred_price:,.2f} Lakhs")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.write("Please check the input values and try again.")

# --- SHAP Explanation ---
if st.checkbox("Show Explanation"):
    try:
        st.subheader("Explanation of Prediction")
        
        # Calculate SHAP values for the single input
        shap_values = explainer.shap_values(input_data)
        
        # The force_plot needs the explainer's expected value
        # We also need to use st.components.v1.html to render the plot in Streamlit
        shap.initjs()
        force_plot_html = shap.force_plot(
            explainer.expected_value,
            shap_values,
            input_data,
            show=False,
            matplotlib=False
        ).html()
        st.components.v1.html(force_plot_html, height=200)

    except Exception as e:
        st.error(f"An error occurred during SHAP explanation: {e}")
        st.write("SHAP explanation could not be generated for the given input.")
