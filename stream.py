import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# load the model

model = tf.keras.models.load_model('model.h5')

# load the encoder and scaler

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('OHE_geography.pkl', 'rb') as file:
    OHE_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# app
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', OHE_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encode = OHE_geo.transform([[geography]]).toarray()
geo_Encoded_df = pd.DataFrame(geo_encode, columns=OHE_geo.get_feature_names_out(['Geography']))


input_data = pd.concat([input_data.reset_index(drop=True), geo_Encoded_df], axis = 1)
# print(input_data)

input_scaled_data = scaler.transform(input_data)
# print(input_scaled_data)

# prediction

prediction = model.predict(input_scaled_data)

st.write(f'Churn Probability: {prediction[0][0] :.2f}')

if prediction[0][0] > 0.5:
    print('The customer is likely to churn')
else:
    print('The customer is not likely to churn')

