import streamlit as st
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend agar tidak butuh display

st.title("💼 Employee Attrition Predictor")
st.write("Aplikasi sederhana untuk memprediksi apakah seorang karyawan akan bertahan atau keluar dari perusahaan.")

# Load Model
try:
    with open("model_tree.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Input fitur
age = st.number_input("Usia Karyawan", min_value=18, max_value=65, step=1)
salary = st.number_input("Gaji (USD/Bulan)", min_value=0.0, step=100.0)
years_at_company = st.number_input("Lama Bekerja (Tahun)", min_value=0.0, step=0.5)
satisfaction = st.slider("Tingkat Kepuasan (%)", 0, 100, 50)

# Siapkan data
features = np.array([[age, salary, years_at_company, satisfaction]])

# Prediksi
if st.button("Prediksi"):
    try:
        if hasattr(model, "predict"):
            prediction = model.predict(features)[0]
        elif callable(model):
            prediction = model(features)[0]
        else:
            raise ValueError("Model tidak memiliki metode prediksi yang valid.")
        st.subheader("Hasil Prediksi")
        st.success(f"📊 Prediksi: *{prediction}*")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
