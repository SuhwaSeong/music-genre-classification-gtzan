import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import base64
import random
import gdown
from io import BytesIO
import tensorflow as tf

# Set Streamlit page config
st.set_page_config(page_title="Music Genre Classifier", layout="centered")

# Session state for refreshing random sample
if "refresh_sample" not in st.session_state:
    st.session_state.refresh_sample = False

# Function to show model accuracy chart
def show_accuracy_chart():
    try:
        acc_rf = pd.read_csv("rf_classification_report.csv", index_col=0).loc["accuracy"].values[0]
        acc_svm = pd.read_csv("svm_classification_report.csv", index_col=0).loc["accuracy"].values[0]
        df_acc = pd.DataFrame({"Model": ["Random Forest", "SVM"], "Accuracy": [acc_rf, acc_svm]})
        st.markdown("### 📈 Model Accuracy Comparison")
        st.bar_chart(df_acc.set_index("Model"))
    except Exception as e:
        st.warning("⚠️ 모델 정확도 그래프를 불러오는 데 실패했습니다.")
        st.exception(e)

# Function to download file from Google Drive if not exists
def download_file_if_missing(file_name, file_id):
    if not os.path.exists(file_name):
        try:
            with st.spinner(f"📅 Downloading {file_name}..."):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)
        except Exception as e:
            st.error(f"❌ 파일 다운로드 실패: {file_name}")
            st.exception(e)
            st.stop()

# Download essential model/report files from Google Drive
files_to_download = {
    "rf_model.pkl": "1oBV5HpsvgoCLr5CYLvrmR6wbMiNP89Gi",
    "svm_model.pkl": "1B3ftW3aIze7gC_QrDK7WAqROBs19jwHt",
    "scaler.pkl": "1tbkqFV95yHrvsLd9NpUvj1QSRpIoen0k",
    "label_encoder.pkl": "1i3wvy68pVMpzjK5y2ny3OeB5KQGEGcQs",
    "rf_classification_report.csv": "1WEkLBZrsFcdFoLLeGH737Feqf5ihmZsB",
    "svm_classification_report.csv": "1FmegZMchjzuX0Tr6aF7rxrvlbmp_Ei-d"
}
for file_name, file_id in files_to_download.items():
    download_file_if_missing(file_name, file_id)

# Load CNN model
@st.cache_resource
def load_cnn_model():
    cnn_model_path = "cnn_genre_model.keras"
    if not os.path.exists(cnn_model_path):
        with st.spinner("Downloading CNN model from Google Drive..."):
            gdown.download("https://drive.google.com/uc?id=1y-OF_0qDIeCj_Cxo3GEYVc4fv_bMu_O2", cnn_model_path, quiet=False)
    model = tf.keras.models.load_model(cnn_model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load classical models (RF, SVM)
def load_model_files(model_name):
    model = joblib.load(f"{model_name.lower()}_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    report_df = pd.read_csv(f"{model_name.lower()}_classification_report.csv", index_col=0)
    return model, scaler, label_encoder, report_df

# Audio feature extraction for classical models
def extract_features(audio_bytes, n_mfcc=13):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).reshape(1, -1), mfcc

# Audio feature extraction for CNN
def extract_mel_spectrogram(audio_bytes, max_len=128):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    pad_width = max(0, max_len - mel_db.shape[1])
    mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant') if pad_width > 0 else mel_db[:, :max_len]
    return mel_db[np.newaxis, ..., np.newaxis], mel_db

# Genre labels for CNN
genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']

# Disable random sample feature on Streamlit Cloud
@st.cache_data
def pick_random_wav_file():
    return None, None

# Audio download link
def get_audio_download_link(file_path, label):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:audio/wav;base64,{b64}" download="test_sample.wav">{label}</a>'

# UI Language selector (English/Korean only shown here for brevity)
lang_dict = {
    "en": {
        "language_name": "English",
        "title": "\ud83c\udfb5 Music Genre Classifier (CNN included)",
        "upload": "Upload a .wav file",
        "select_model": "Select model",
        "prediction": "\ud83c\udfb6 Predicted genre:",
        "prob": "### \ud83d\udd0d Prediction Probabilities",
        "mfcc": "Show MFCC Heatmap",
        "mel": "Show Mel Spectrogram",
        "mfcc_title": "MFCC Features",
        "mel_title": "Mel Spectrogram",
        "upload_prompt": "Please upload a .wav file to begin."
    },
    "ko": {
        "language_name": "Korean (\ud55c\uad6d\uc5b4)",
        "title": "\ud83c\udfb5 \uc74c\uc545 \uc7a5\ub974 \ubd84\ub958\uae30 (CNN \ud3ec\ud568)",
        "upload": ".wav \ud30c\uc77c\uc744 \uc5c5\ub85c\ub4dc\ud558\uc138\uc694",
        "select_model": "\ubaa8\ub378 \uc120\ud0dd",
        "prediction": "\ud83c\udfb6 \uc608\uce21\ub41c \uc7a5\ub974:",
        "prob": "### \ud83d\udd0d \uc608\uce21 \ud655\ub960",
        "mfcc": "MFCC \ud788\ud2b8\ub9f5 \ubcf4\uae30",
        "mel": "Mel \uc2a4\ud398\ud06c\ud1a0\uadf8\ub7a8 \ubcf4\uae30",
        "mfcc_title": "MFCC \ud2b9\uc9d5",
        "mel_title": "Mel \uc2a4\ud398\ud06c\ud1a0\uadf8\ub7a8",
        "upload_prompt": ".wav \ud30c\uc77c\uc744 \uc5c5\ub85c\ub4dc\ud574\uc8fc\uc138\uc694."
    }
}

# UI
selected_lang = st.sidebar.selectbox("Language / \uc5b8\uc5b4", options=list(lang_dict.keys()), format_func=lambda x: lang_dict[x]["language_name"])
texts = lang_dict[selected_lang]

st.title(texts["title"])
model_option = st.selectbox(texts["select_model"], ["Random Forest", "SVM", "CNN"])

uploaded_file = st.file_uploader(texts["upload"], type=["wav"])
if uploaded_file:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")
    
    if model_option == "CNN":
        model = load_cnn_model()
        features, mel = extract_mel_spectrogram(audio_bytes)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_label = genre_labels[predicted_index]

        st.success(f"{texts['prediction']} `{predicted_label}`")
        st.markdown(texts["prob"])
        st.bar_chart(dict(zip(genre_labels, prediction[0])))
        if st.checkbox(texts["mel"]):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mel, cmap="YlGnBu", ax=ax)
            ax.set(title=texts["mel_title"], xlabel="Time", ylabel="Mel Bands")
            st.pyplot(fig)
    else:
        model, scaler, label_encoder, _ = load_model_files(model_option)
        features, mfcc = extract_features(audio_bytes)
        features_scaled = scaler.transform(features)
        pred_encoded = model.predict(features_scaled)
        pred_label = label_encoder.inverse_transform(pred_encoded)

        st.success(f"{texts['prediction']} `{pred_label[0]}`")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            class_labels = label_encoder.classes_
            st.markdown(texts["prob"])
            st.bar_chart(dict(zip(class_labels, proba)))
        if st.checkbox(texts["mfcc"]):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
            ax.set(title=texts["mfcc_title"], xlabel="Time", ylabel="MFCC Coefficients")
            st.pyplot(fig)
else:
    st.info(texts["upload_prompt"])
