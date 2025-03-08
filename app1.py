import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from geopy.distance import geodesic

model=joblib.load("credit_card_fraud_detect_model.jb")
encoder=joblib.load('label_encoder.jb')

def havedistance(lat1 ,long1,lat2,long2):
    return geodesic((lat1, long1), (lat2, long2)).km


st.title("Credit Card Fraud Detection")
st.write("Enter the transaction details below:")

merch = st.text_input("Merchant name", 'fraud_Rippin, Kub and Mann')
category = st.text_input("Category", 'misc_net')
cc_num = st.text_input('Credit Card Number', '703186189652095')

amt = st.number_input("Transaction Amount", min_value=0.0, value=4.97, format='%.2f')
lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=36.0788, format='%.6f')
long = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-81.1781, format='%.6f')
merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=36.011293, format='%.6f')
merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-82.048315, format='%.6f')

hour = st.slider("Transaction Hour", 0, 23, 0)
month = st.slider("Transaction Month", 1, 12, 1)
day = st.slider("Transaction Day", 1, 31, 1)

gender = st.selectbox('Gender', ['Female', 'Male'])

distance=havedistance(lat,long,merch_lat,merch_long)
if st.button('Check for fraud'):
    if merch and category and cc_num :
        input_data = pd.DataFrame([[merch, category,amt,distance,hour,day,month,gender, cc_num]],columns=['merchant','category','amt','distance','hour','day','month','gender','cc_num'])
        categorical_col = ['merchant','category','gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col]=-1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x:hash(x) % (10 ** 2))
        prediction = model.predict(input_data)[0]
        result = "Fraudulant Transaction" if prediction == 1 else " Legitimate Transaction"
        st.subheader(f"Prediction: {result}")
    else:
        st.error("Please Fill all required fields")
        
        
