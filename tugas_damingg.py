import streamlit as st
import pickle
import numpy as np

# Judul Aplikasi
st.title("🌸 Iris Flower Classifier 🌸")
st.write("Aplikasi sederhana untuk memprediksi jenis bunga iris menggunakan model scikit-learn.")

# Load Model
try:
    with open("model_sklearn.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Input fitur pengguna
st.header("Masukkan Fitur Bunga")
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Data input dalam bentuk array
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Tombol prediksi
if st.button("Prediksi"):
    try:
        prediction = model.predict(features)[0]
        st.subheader("Hasil Prediksi 🌼")
        st.success(f"Jenis Bunga: **{prediction}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
