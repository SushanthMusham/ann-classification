
import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load the encoders and scaler
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Churn Prediction")

geography = st.selectbox(
    "Geography",
    ["France", "Spain", "Germany"],
    index=None,
    placeholder="Select a geography"
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
    index=None,
    placeholder="Select a gender"
)

age = st.slider("Age", 18, 100, 25)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
credit_score = st.number_input("Credit Score", min_value=0, value=600)
estimated_salary = st.number_input(
    "Estimated Salary", min_value=0.0, value=50000.0
)
tenure = st.slider("Tenure", 0, 10, 3)
num_of_products = st.slider("Number of Products", 1, 4, 1)

has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict Churn"):

    if geography is None or gender is None:
        st.warning("Please select both geography and gender.")

    else:
        # Create input DataFrame using training feature names
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [
                label_encoder_gender.transform([gender])[0]
            ],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        })

        # One-hot encode Geography
        geo_encoded = onehot_encoder_geo.transform(
            pd.DataFrame({"Geography": [geography]})
        )

        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(
                ["Geography"]
            )
        )

        # Combine features
        input_data = pd.concat(
            [input_data, geo_encoded_df],
            axis=1
        )

        # Ensure the feature order matches the training data.
        # Replace this list if your actual training order differs.
        expected_columns = list(scaler.feature_names_in_)
        input_data = input_data[expected_columns]

        # Scale and predict
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled, verbose=0)
        probability = float(prediction[0][0])

        st.metric("Churn Probability", f"{probability:.2%}")

        if probability >= 0.5:
            st.error("Prediction: Churn")
        else:
            st.success("Prediction: No Churn")
