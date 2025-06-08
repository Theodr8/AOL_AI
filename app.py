
import os
import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.image import resize

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Genre Musik", layout="wide")
st.title("🎵 Prediksi Genre Musik")
st.write("Unggah file audio (.wav, .mp3, .m4a), dan AI akan menganalisis genrenya.")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model_path = "Trained_model.h5"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- List Genre ---
genre = ['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop']

# --- Preprocessing Audio ---
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return np.array(data)

    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)

    if len(audio_data) < chunk_samples:
        num_chunks = 1
    else:
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:min(end, len(audio_data))]
        if len(chunk) == 0:
            continue
        try:
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            if mel_spectrogram.ndim == 2:
                mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
            if mel_spectrogram.shape[-1] != 1:
                mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
            mel_resized = resize(mel_spectrogram, target_shape)
            data.append(mel_resized)
        except Exception as e:
            st.warning(f"Chunk {i} gagal: {e}")
            continue
    return np.array(data)

# --- Prediksi Genre ---
def prediksi(tes_x, model):
    if model is None:
        return None, {"error": "Model tidak dimuat."}
    if tes_x.shape[0] == 0:
        return None, {"error": "Data audio kosong."}
    try:
        y_pred = model.predict(tes_x)
        genre_musik = np.argmax(y_pred, axis=1)
        total = len(genre_musik)
        count = [0] * len(genre)
        for i in genre_musik:
            if i < len(genre):
                count[i] += 1
        persentase = [(c / total) * 100 for c in count]
        max_count = max(count)
        max_genres = [genre[i] for i, c in enumerate(count) if c == max_count]
        hasil = {
            "genre_terbanyak": max_genres,
            "jumlah_per_genre": {genre[i]: count[i] for i in range(len(genre))},
            "persentase_per_genre": {genre[i]: round(persentase[i], 2) for i in range(len(genre))}
        }
        return np.argmax(persentase), hasil
    except Exception as e:
        return None, {"error": f"Prediksi gagal: {e}"}

# --- Main ---
model = load_model()

if model:
    uploaded_file = st.file_uploader("Unggah file audio", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("🔍 Prediksi Genre"):
            with st.spinner("Memproses..."):
                # Simpan sementara file
                os.makedirs("uploads", exist_ok=True)
                temp_path = os.path.join("uploads", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Proses dan prediksi
                tes_x = load_and_preprocess_data(temp_path)
                prediksi_genre_index, hasil = prediksi(tes_x, model)

                if hasil and "error" not in hasil:
                    st.success(f"🎶 Genre Dominan: **{genre[prediksi_genre_index].capitalize()}**")
                    df = pd.DataFrame(
                        list(hasil["persentase_per_genre"].items()),
                        columns=["Genre", "Persentase (%)"]
                    ).sort_values("Persentase (%)", ascending=False).set_index("Genre")
                    st.bar_chart(df)
                    st.dataframe(df)
                else:
                    st.error(hasil["error"])
                os.remove(temp_path)
else:
    st.error("Model tidak tersedia. Periksa path dan format.")
