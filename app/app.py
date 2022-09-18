import streamlit as st
import numpy as np
import pandas as pd
import pickle
from utils import preprocess_data, columns

# Dictonaries for text and values mapping
gender_dict = {0:'Male', 1:'Female'}
def gender_format(option):
    return gender_dict[option]

cp_dict = {0:'Typical Angina', 1:'ATypical Angina', 2:'Non-Anginal Pain', 3:'Asymptomatic'}
def cp_format(option):
    return cp_dict[option]

bool_dict = {0:'False', 1:'True'}
def bool_format(option):
    return bool_dict[option]

restecg_dict = {0:'Normal', 1:'Wave Normality', 2:'Left Ventricular Hypertrophy'}    
def restecg_format(option):
    return restecg_dict[option]

st.title('Heart Attack Prediction')

age = st.text_input('Patient Age', '60')
sex = st.selectbox('Patient Gender', options=list(gender_dict.keys()), format_func = gender_format)
cp = st.selectbox('Chest Pain Type', options=list(cp_dict.keys()), format_func = cp_format)
trtbps = st.text_input('Resting Blood Pressure (in mm Hg)', '110')
chol = st.text_input('Cholestrol in mg/dl', '60')
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl ?', options=list(bool_dict.keys()), format_func = bool_format)
restecg = st.selectbox('Resting Electrocardiographic Results', options=list(restecg_dict.keys()), format_func=restecg_format)
thalachh = st.text_input('Maximum Heart Rate Achieved', '140')
exng = st.radio('Exercise Induced Angina', options=list(bool_dict.keys()), format_func = bool_format)
oldpeak = st.text_input('Previous Peak', '140')
slp = st.text_input('Slope', '100')
caa = st.text_input('Number of Major Vessels', '2')
thall = st.selectbox('Thalium Stress Test Result', ('0', '1', '2', '3'))

def predict():
    """Grab input, pass it to preprocessing and then return prediction"""
    # Dump input to a df
    # row = np.array([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall])
    # X = pd.DataFrame([row], columns = columns)
    row = np.array([sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall])
    X = pd.DataFrame([row], columns = columns)

    # Scale data
    X = preprocess_data(X)

    # Load model and predict
    model = pickle.load(open('../model/model.sav', 'rb'))
    prediction = model.predict(X)[0]

    # Check prediction
    # if prediction == 1:
    #     st.error('Patient is likely have a heart attack')
    # else:
    #     st.success('Patient is not likely to have a heart attack')
    st.success('Patient is not likely to have a heart attack')

st.button('Predict', on_click = predict)
