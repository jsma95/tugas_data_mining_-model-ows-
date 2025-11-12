import streamlit as st
import pickle
import numpy as np

st.title("💼 Employee Data Prediction App")
st.write("Aplikasi sederhana untuk memprediksi data karyawan menggunakan model dari Orange.")

# Load model
try:
    with open("model_data_mining.pkcls", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Input fitur (contoh untuk dataset karyawan)
age = st.number_input("Umur (tahun)", min_value=18, max_value=65, step=1)
education = st.number_input("Tingkat Pendidikan (1-5)", min_value=1, max_value=5, step=1)
experience = st.number_input("Lama Pengalaman Kerja (tahun)", min_value=0, max_value=40, step=1)
salary = st.number_input("Gaji Saat Ini (juta rupiah)", min_value=0.0, max_value=100.0, step=0.5)

# Siapkan data untuk prediksi
features = np.array([[age, education, experience, salary]])

# Tombol prediksi
if st.button("🔍 Prediksi"):
    try:
        prediction = model(features)[0]
        st.subheader("Hasil Prediksi")
        st.success(f"📊 Hasil model: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memproses prediksi: {e}")
