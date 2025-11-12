import streamlit as st
import pickle
import numpy as np

# === Load Model ===
try:
    with open("model_data_mining.pkcls", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# === Judul Aplikasi ===
st.title("💼 Employee Attrition Predictor")
st.write("Aplikasi sederhana untuk memprediksi kemungkinan karyawan bertahan atau keluar menggunakan model dari Orange.")

# === Input fitur (ubah sesuai dataset employee_data.csv kamu) ===
age = st.number_input("Umur Karyawan", min_value=18, max_value=65, step=1)
salary = st.number_input("Gaji Bulanan", min_value=0.0, step=100.0)
years_at_company = st.number_input("Lama Bekerja (tahun)", min_value=0.0, max_value=40.0, step=0.5)
satisfaction = st.slider("Tingkat Kepuasan Kerja", 0.0, 1.0, 0.5)

# === Siapkan data untuk prediksi ===
features = np.array([[age, salary, years_at_company, satisfaction]])

# === Prediksi ===
if st.button("Prediksi"):
    try:
        prediction = model(features)[0]  # untuk model Orange
        st.subheader("📊 Hasil Prediksi")
        st.success(f"Hasil model: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
