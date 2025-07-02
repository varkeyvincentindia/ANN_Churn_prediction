import streamlit as st 
import numpy as np 
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

#load training model

model= tf.keras.models.load_model('model.h5')

# Load th encoders and scalars

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender= pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geo= pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar= pickle.load(file)

## streamlit app

st.title('Customer churn prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

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

geo_column_output= onehot_encoder_geo.transform([[geography]]).toarray()

geo_out= onehot_encoder_geo.get_feature_names_out(['Geography'])

new_geo_columns=pd.DataFrame(geo_column_output, columns=geo_out)
#new_geo_columns

## Combine all the one hot encoded colums
input_data= pd.concat([input_data.reset_index(drop=True) ,new_geo_columns], axis=1)

#scale the data

input_data_scaled= scalar.transform(input_data)

prediction= model.predict(input_data_scaled)
prediction_probability= prediction[0][0]

st.write(f'Churn rate= {prediction_probability*100}%')