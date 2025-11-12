import streamlit as st
import pickle
import numpy as np
import os

st.title("🧠 Aplikasi Prediksi Data Mining")
st.write("Aplikasi ini memuat model dari Orange (.pkcls) dan menampilkannya di Streamlit.")

MODEL_ORANGE = "mode_data_mining.pkcls"
MODEL_SKLEARN = "model_sklearn.pkl"

# --- Fungsi konversi model Orange ke sklearn
def convert_orange_to_sklearn():
    try:
        with open(MODEL_ORANGE, "rb") as f:
            orange_model = pickle.load(f)

        # Ambil model scikit-learn dari Orange
        skl_model = getattr(orange_model, "skl_model", None)
        if skl_model is None:
            st.error("❌ Model Orange tidak berbasis scikit-learn, tidak bisa dikonversi.")
            return None

        with open(MODEL_SKLEARN, "wb") as f:
            pickle.dump(skl_model, f)

        st.success("✅ Model Orange berhasil dikonversi menjadi model_sklearn.pkl")
        return skl_model

    except Exception as e:
        st.error(f"❌ Gagal konversi model Orange: {e}")
        return None

# --- Load model (prioritaskan model sklearn)
model = None
if os.path.exists(MODEL_SKLEARN):
    with open(MODEL_SKLEARN, "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model sklearn berhasil dimuat.")
elif os.path.exists(MODEL_ORANGE):
    st.warning("⚠️ File model Orange ditemukan, mencoba konversi ke sklearn...")
    model = convert_orange_to_sklearn()
else:
    st.error("❌ Tidak ditemukan file 'mode_data_mining.pkcls' atau 'model_sklearn.pkl'.")

# --- Jika model sudah siap
if model:
    st.subheader("Masukkan Nilai Fitur")

    # Ubah sesuai jumlah kolom dataset kamu
    fitur1 = st.number_input("Fitur 1", value=0.0)
    fitur2 = st.number_input("Fitur 2", value=0.0)
    fitur3 = st.number_input("Fitur 3", value=0.0)
    fitur4 = st.number_input("Fitur 4", value=0.0)

    data = np.array([[fitur1, fitur2, fitur3, fitur4]])

    if st.button("🔮 Prediksi"):
        try:
            pred = model.predict(data)
            st.subheader("🎯 Hasil Prediksi")
            st.success(f"Hasil model: **{pred[0]}**")
        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")
