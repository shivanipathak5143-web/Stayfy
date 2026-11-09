import pickle

import pandas as pd
import streamlit as st

# Show the title immediately so the page is never blank while loading
st.title("Customer Churn Prediction")


@st.cache_resource
def load_artifacts():
    # Heavy TensorFlow import happens once, inside the cache
    import tensorflow as tf

    model = tf.keras.models.load_model("model.h5", compile=False)

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geography.pkl", "rb") as f:
        onehot_encoder_geography = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geography, scaler


with st.spinner("Loading model... (first load can take a minute)"):
    model, label_encoder_gender, onehot_encoder_geography, scaler = load_artifacts()

# ---------- User inputs ----------
geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 40)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
credit_score = st.number_input("Credit Score", min_value=0.0, value=600.0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
tenure = st.slider("Tenure", 0, 10, 3)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# ---------- Build the model input ----------
input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)

# Pass a DataFrame with the column name so sklearn doesn't warn about feature names
geo_encoded = onehot_encoder_geography.transform(
    pd.DataFrame([[geography]], columns=["Geography"])
)
if hasattr(geo_encoded, "toarray"):  # sparse output -> dense
    geo_encoded = geo_encoded.toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geography.get_feature_names_out(["Geography"]),
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Make sure columns are in the exact order the scaler was fitted on
if hasattr(scaler, "feature_names_in_"):
    input_data = input_data[list(scaler.feature_names_in_)]

input_data_scaled = scaler.transform(input_data)

# ---------- Predict ----------
prediction = model.predict(input_data_scaled, verbose=0)
prediction_prob = float(prediction[0][0])

st.subheader("Result")
if prediction_prob > 0.5:
    st.error(f"The customer is likely to churn (probability: {prediction_prob:.2f}).")
else:
    st.success(f"The customer is unlikely to churn (probability: {prediction_prob:.2f}).")