import streamlit as st
import pickle
import numpy as np

st.title("💼 Employee Data Prediction App")
st.write("Aplikasi sederhana untuk memprediksi data karyawan menggunakan model scikit-learn.")

# Load model
try:
    with open("model_sklearn.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Input fitur
age = st.number_input("Umur (tahun)", 18, 65, 25)
education = st.number_input("Tingkat Pendidikan (1-5)", 1, 5, 3)
experience = st.number_input("Pengalaman Kerja (tahun)", 0, 40, 5)
salary = st.number_input("Gaji Saat Ini (juta rupiah)", 0.0, 100.0, 10.0, 0.5)

# Siapkan data untuk prediksi
features = np.array([[age, education, experience, salary]])

# Tombol prediksi
if st.button("🔍 Prediksi"):
    try:
        prediction = model.predict(features)[0]
        st.subheader("Hasil Prediksi")
        st.success(f"📊 Prediksi Model: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Gagal melakukan prediksi: {e}")
