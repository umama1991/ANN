import streamlit as st 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler,OneHotEncoder , LabelEncoder
import pickle

model=tf.keras.models.load_model("Ann_model.h5")

with open("onehot_encoder_geo.pkl","rb")as file:
    onehot_encoder_geo=pickle.load(file)
with open("label_encoder_gender.pkl","rb")as file:
    label_encoder_gender=pickle.load(file)
with open("scaler.pkl","rb")as file:
    scaler=pickle.load(file)
#streamlit
st.title("Customer Churn Prediction")
#user Input
Geography=st.selectbox("Geography",onehot_encoder_geo.category_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age= st.slider("Age",18,92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_of_products= st.slider("No Of Products",1,4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.sselectbox("Is Active Member",[0,1])
#prepare the input data
input_data=pd.DataFrame({
    
    "CreditScore":[credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Tenure":[tenure],
    "Age" :[Age],
    "Balance" : [Balance],
    "NumOfProducts" : [num_of_products],
    "HasCrCard" :[has_cr_card],
    "IsActiveMember" : [is_active_member],
    "EstimatedSalary" : [estimated_salary]
})
geo_encoded=onehot_encoder_geo.transform([Geography]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_name)
# combine onehot encoder with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled =  scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")


if prediction_proba > 0.5:
    st.write("The Customer is likely to leave the bank")
else:
    st.write('The Customer will not leave the bank')






