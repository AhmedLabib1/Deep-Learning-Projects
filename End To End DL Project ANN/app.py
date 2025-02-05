import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Load the saved model and encoders
model = tf.keras.models.load_model('model.h5')

with open('Pickled Objects/One_Hot_Encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('Pickled Objects/Label_Encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('Pickled Objects/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title('Customer Churn Prediction')
st.write("Enter customer details to predict churn.")

# User input fields
city = st.text_input("City")
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
total_charges = st.number_input("Total Charges", min_value=0.0, format='%.2f')

# Preprocess user input
def preprocess_input():
    user_data = pd.DataFrame([[city, gender, senior_citizen, partner, dependents, phone_service, paperless_billing, 
                               multiple_lines, internet_service, online_security, contract, payment_method, 
                               total_charges]],
                             columns=['City', 'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing',
                                      'Multiple Lines', 'Internet Service', 'Online Security', 'Contract', 'Payment Method',
                                      'Total Charges'])
    
    # Apply label encoding
    for feature in ['City', 'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']:
        user_data[feature] = label_encoder.transform(user_data[feature])
    
    # Apply one-hot encoding
    encoded_data = one_hot_encoder.transform(user_data[['Multiple Lines', 'Internet Service', 'Online Security', 'Contract', 'Payment Method']])
    encoded_columns = one_hot_encoder.get_feature_names_out(['Multiple Lines', 'Internet Service', 'Online Security', 'Contract', 'Payment Method'])
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
    user_data = pd.concat([user_data.drop(columns=['Multiple Lines', 'Internet Service', 'Online Security', 'Contract', 'Payment Method']), encoded_df], axis=1)
    
    # Apply scaling
    scaled_data = scaler.transform(user_data)
    return scaled_data

if st.button("Predict Churn"):
    processed_input = preprocess_input()
    prediction = model.predict(processed_input)
    result = "Churn" if prediction[0] > 0.5 else "No Churn"
    st.write(f"Prediction: {result}")
