# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:19:43 2023

@author: TanveerKader
"""

import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#file1 = open('C:/Users/TanveerKader/Desktop/Ml Web app/sales-prediction-webapp/softdrinks_sale_lr_model.sav', 'rb')
#file2 = open('C:/Users/TanveerKader/Desktop/Ml Web app/sales-prediction-webapp/softdrinks_sale_xgb_model.sav', 'rb')
#file3 = open('C:/Users/TanveerKader/Desktop/Ml Web app/sales-prediction-webapp/icecream_sale_lr_model.sav', 'rb')
#file4 = open('C:/Users/TanveerKader/Desktop/Ml Web app/sales-prediction-webapp/icecream_sale_xgb_model.sav', 'rb')

file1 = open('softdrinks_sale_lr_model.sav', 'rb')
file2 = open('softdrinks_sale_xgb_model.sav', 'rb')
file3 = open('icecream_sale_lr_model.sav', 'rb')
file4 = open('icecream_sale_xgb_model.sav', 'rb')

lr_model_sd = pickle.load(file1) 
xgb_model_sd = pickle.load(file2) 

lr_model_ic = pickle.load(file3) 
xgb_model_ic = pickle.load(file4) 


def softdrinks_sale_prediction(input_data):

    # change input data to nympy array
    input_data_arr = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_arr.reshape(1, -1)

    prediction_lr = lr_model_sd.predict(input_data_reshaped)
    prediction_xgb = xgb_model_sd.predict(input_data_reshaped)
    
    
    return prediction_lr[0], prediction_xgb[0]

def icecream_sale_prediction(input_data):

    # change input data to nympy array
    input_data_arr = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_arr.reshape(1, -1)

    prediction_lr = lr_model_ic.predict(input_data_reshaped)
    prediction_xgb = xgb_model_ic.predict(input_data_reshaped)
    
    
    return prediction_lr[0], prediction_xgb[0]


#sidebar
with st.sidebar:
    selected = option_menu("Sales Prediction System",
                           
                           [ "Softdrinks Sales Prediction",
                            "Icecream Sales Prediction"],
                           
                           icons = ["cup-straw", "snow"],
                           
                           default_index = 0)

# Soft drinks prediction 
if(selected == "Softdrinks Sales Prediction"):
    #title
    st.title("Softdrinks Sales Prediction")
    temperature = st.text_input("Enter the temperature today (C)")
    
    result_lr = ''
    result_xgb = ''
    
    #button
    if st.button("Predict"):
        result_lr, result_xgb = softdrinks_sale_prediction([float(temperature)])
        
    st.subheader("Todays Sales Prediction (Litres)")
    st.caption("Linear Regression Model")
    st.success(result_lr)
    st.caption("XGBoost Regressor Model")
    st.success(result_xgb)
    


if(selected == "Icecream Sales Prediction"):
    #title
    st.title("Icecream Sales Prediction")
    temperature = st.text_input("Enter the temperature today (C)")
    
    result_lr = ''
    result_xgb = ''
    
    #button
    if st.button("Predict"):
        result_lr, result_xgb = softdrinks_sale_prediction([float(temperature)])
        
    st.subheader("Todays Sales Prediction (Litres)")
    st.caption("Linear Regression Model")
    st.success(result_lr)
    st.caption("XGBoost Regressor Model")
    st.success(result_xgb)
    
#streamlit run "C:/Users/TanveerKader/Desktop/Ml Web app/sales-prediction-webapp/salesprediction.py"