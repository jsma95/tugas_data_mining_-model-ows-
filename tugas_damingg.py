import streamlit as st
import pickle
import numpy as np

# Load Model
with open("model-iris.pkcls", "rb") as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Classifier 🌸")
st.write("Aplikasi sederhana untuk memprediksi jenis bunga iris menggunakan model Neural Network dari Orange.")

# Input features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Prepare data
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict
if st.button("Prediksi"):
    prediction = model(features)[0]  # Orange model returns something like ["Iris-setosa"]
    
    st.subheader("Hasil Prediksi")
    st.success(f"🌼 Jenis Bunga: *{prediction}*")
