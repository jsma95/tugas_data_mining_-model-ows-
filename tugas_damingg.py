import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os 

st.set_page_config(
    page_title="Prediksi Departemen Karyawan",
    page_icon="",
    layout="wide"
)


MODEL_FILE_NAME = 'ModelNeuralNetwork.pkcls'

@st.cache_resource
def load_model(file_path):
    """
    Memuat model pickle dari file.
    Menggunakan cache agar model hanya dimuat sekali saat aplikasi dimulai.
    """
    try:
        with open(file_path, 'rb') as f:
            # Memuat model
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: File model '{file_path}' tidak ditemukan.")
        st.info("Pastikan file 'ModelNeuralNetwork.pkcls' berada di direktori yang sama dengan script Streamlit Anda.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.info("Pastikah library yang diperlukan (Orange3, scikit-learn) sudah terinstal.")
        return None

model = load_model(MODEL_FILE_NAME)

st.title("Aplikasi Prediksi Departemen Karyawan")
st.write(
    "Aplikasi ini menggunakan model Neural Network yang telah dilatih "
    "untuk memprediksi departemen karyawan."
)

if model:
    try:
        feature_names = [attr.name for attr in model.domain.attributes]
        class_names = model.domain.class_var.values
    except Exception as e:
        st.error(f"Gagal membaca metadata dari model (nama fitur/kelas): {e}")
        st.stop() 

    st.divider()

    st.sidebar.header("Input Fitur Karyawan")
    st.sidebar.write("Silakan masukkan data karyawan untuk diprediksi:")

    st.info(f"Model ini membutuhkan 3 fitur: **{', '.join(feature_names)}**")

    salary = st.sidebar.number_input(
        f"Input: {feature_names[0]} (Gaji IDR)",
        min_value=1000000,
        value=10000000,
        step=100000,
        help="Masukkan gaji dalam IDR (misal: 10000000)"
    )
    
    performance = st.sidebar.number_input(
        f"Input: {feature_names[1]} (Skor Kinerja)",
        min_value=1.0,
        max_value=5.0,
        value=3.5,
        step=0.1,
        help="Masukkan skor kinerja (rentang 1.0 - 5.0)"
    )
    
    experience = st.sidebar.number_input(
        f"Input: {feature_names[2]} (Pengalaman Tahun)",
        min_value=0,
        max_value=40,
        value=5,
        step=1,
        help="Masukkan lama pengalaman dalam tahun (misal: 5)"
    )

    if st.sidebar.button("Prediksi Departemen", use_container_width=True):

        input_data = np.array([[
            salary, 
            performance, 
            experience
        ]])

        try:
            predictions, probabilities = model(input_data)
            
            prediction_index = int(predictions[0])
            predicted_class = class_names[prediction_index]
            probability_values = probabilities[0] # Ambil array probabilitas

            st.subheader("Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Prediksi Departemen", value=predicted_class)
            with col2:
                st.metric(label="Tingkat Keyakinan", value=f"{probability_values[prediction_index]:.2%}")

            st.divider()

            st.subheader("Probabilitas per Departemen")

            df_probs = pd.DataFrame({
                'Departemen': class_names,
                'Probabilitas': probability_values
            })
            
            df_probs['Probabilitas'] = df_probs['Probabilitas'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(df_probs.set_index('Departemen'), use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}")
            st.error("Pastikan model ('ModelNeuralNetwork.pkcls') kompatibel dan datanya benar.")
else:
    st.warning("Model tidak dapat dimuat. Aplikasi tidak dapat berjalan.")
