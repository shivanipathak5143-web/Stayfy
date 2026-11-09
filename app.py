import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


