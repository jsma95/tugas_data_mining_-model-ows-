import streamlit as st
import pickle
import numpy as np

st.title("💼 Employee Data Classifier")
st.write("Aplikasi sederhana untuk memprediksi status karyawan menggunakan model Decision Tree dari scikit-learn.")

# Load model
try:
    with open("model_sklearn.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Input fitur (ubah sesuai dataset kamu)
age = st.number_input("Usia Karyawan", min_value=18, max_value=65, step=1)
experience = st.number_input("Pengalaman Kerja (tahun)", min_value=0, max_value=40, step=1)
salary = st.number_input("Gaji (juta)", min_value=0.0, max_value=100.0, step=0.5)

# Gabungkan input jadi array
features = np.array([[age, experience, salary]])

# Prediksi
if st.button("Prediksi"):
    try:
        prediction = model.predict(features)[0]
        st.subheader("Hasil Prediksi")
        st.success(f"📊 Hasil model: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat prediksi: {e}")
