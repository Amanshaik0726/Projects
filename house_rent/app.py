import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("house_rent.h5")


# Define the main function for the Streamlit app
def main():
    st.title("House Rent Prediction ")
    st.write("Enter House Details to Predict Rent")

    # Input fields for user to input house details
    num_bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)
    house_size = st.number_input("Size of the House", min_value=100, max_value=10000, step=10)
    area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
    area_type_code = {"Super Area": 1, "Carpet Area": 2, "Built Area": 3}
    pin_code = st.number_input("Pin Code of the City", min_value=100, max_value=999999, step=1)
    furnishing_status = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
    furnishing_code = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
    tenant_type = st.selectbox("Tenant Type", ["Bachelors", "Bachelors/Family", "Only Family"])
    tenant_code = {"Bachelors": 1, "Bachelors/Family": 2, "Only Family": 3}
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)

    # Convert user inputs into features array
    features = np.array([[num_bhk, house_size, area_type_code[area_type], pin_code, furnishing_code[furnishing_status], tenant_code[tenant_type], num_bathrooms]])

    # Predict house price
    if st.button("Predict Rent"):
        predicted_rent = model.predict(features) 
        st.write(f"Predicted House Rent: {predicted_rent}")

if __name__ == "__main__":
    main()
