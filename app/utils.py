from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pickle

# Data cols
columns = [
    #'age',
    'sex',
    'cp',
    'trtbps',
    'chol',
    'fbs',
    'restecg',
    'thalachh',
    'exng',
    'oldpeak',
    'slp',
    'caa',
    'thall'
]

def preprocess_data(data):
    """Preprocess input to be suitable for model prediction"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data
