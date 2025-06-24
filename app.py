import streamlit as st
from googletrans import Translator
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import base64
import gdown
from io import BytesIO
import tensorflow as tf

# 페이지 설정
st.set_page_config(page_title="Music Genre Classifier", layout="centered")

# 번역 함수
def translate_text(text, dest_lang="en"):
    translator = Translator()
    try:
        translated = translator.translate(text, dest=dest_lang)
        return translated.text
    except Exception as e:
        st.warning(f"⚠️ 번역 실패: {e}")
        return text

# 언어 선택
lang = st.sidebar.selectbox("Select language", ["en", "ko", "de", "fr", "es"])

# 초기 안내
st.sidebar.info(translate_text("Please select a language and upload a .wav file.", lang))

# 모델 정확도 시각화
def show_accuracy_chart():
    try:
        acc_rf = pd.read_csv("rf_classification_report.csv", index_col=0).loc["accuracy"].values[0]
        acc_svm = pd.read_csv("svm_classification_report.csv", index_col=0).loc["accuracy"].values[0]
        df_acc = pd.DataFrame({"Model": ["Random Forest", "SVM"], "Accuracy": [acc_rf, acc_svm]})
        st.markdown(translate_text("### 📊 Model Accuracy Comparison", lang))
        st.bar_chart(df_acc.set_index("Model"))
    except Exception as e:
        st.warning(translate_text("⚠️ Failed to load model accuracy chart.", lang))
        st.exception(e)

# 파일 다운로드 함수
def download_file_if_missing(file_name, file_id):
    if not os.path.exists(file_name):
        try:
            with st.spinner(f"📃 Downloading {file_name}..."):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)
        except Exception as e:
            st.error(f"❌ Failed to download: {file_name}")
            st.exception(e)
            st.stop()

# 필요한 파일 다운로드
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

# 모델 불러오기
@st.cache_resource
def load_cnn_model():
    cnn_model_path = "cnn_genre_model.keras"
    if not os.path.exists(cnn_model_path):
        with st.spinner("Downloading CNN model from Google Drive..."):
            gdown.download("https://drive.google.com/uc?id=1y-OF_0qDIeCj_Cxo3GEYVc4fv_bMu_O2", cnn_model_path, quiet=False)
    model = tf.keras.models.load_model(cnn_model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_files(model_name):
    model = joblib.load(f"{model_name.lower()}_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

# 특징 추출 함수
def extract_features(audio_bytes, n_mfcc=13):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).reshape(1, -1), mfcc

def extract_mel_spectrogram(audio_bytes, max_len=128):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    pad_width = max(0, max_len - mel_db.shape[1])
    mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant') if pad_width > 0 else mel_db[:, :max_len]
    return mel_db[np.newaxis, ..., np.newaxis], mel_db

# 장르 라벨
genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']

# 제목
st.title(translate_text("🎵 Music Genre Classifier (CNN included)", lang))
model_option = st.selectbox(translate_text("Select model", lang), ["Random Forest", "SVM", "CNN"])
uploaded_file = st.file_uploader(translate_text("Upload a .wav file", lang), type=["wav"])

# 예측
if uploaded_file:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

    if model_option == "CNN":
        model = load_cnn_model()
        features, mel = extract_mel_spectrogram(audio_bytes)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_label = genre_labels[predicted_index]

        st.success(f"{translate_text('Predicted genre:', lang)} `{predicted_label}`")
        st.markdown(translate_text("### 🔍 Prediction Probabilities", lang))
        st.bar_chart(dict(zip(genre_labels, prediction[0])))

        if st.checkbox(translate_text("Show Mel Spectrogram", lang)):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mel, cmap="YlGnBu", ax=ax)
            ax.set(title=translate_text("Mel Spectrogram", lang), xlabel="Time", ylabel="Mel Bands")
            st.pyplot(fig)
            plt.close(fig)

    else:
        model, scaler, label_encoder = load_model_files(model_option)
        features, mfcc = extract_features(audio_bytes)
        features_scaled = scaler.transform(features)
        pred_encoded = model.predict(features_scaled)
        pred_label = label_encoder.inverse_transform(pred_encoded)

        st.success(f"{translate_text('Predicted genre:', lang)} `{pred_label[0]}`")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            class_labels = label_encoder.classes_
            st.markdown(translate_text("### 🔍 Prediction Probabilities", lang))
            st.bar_chart(dict(zip(class_labels, proba)))

        if st.checkbox(translate_text("Show MFCC Heatmap", lang)):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
            ax.set(title=translate_text("MFCC Features", lang), xlabel="Time", ylabel="MFCC Coefficients")
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info(translate_text("📂 Please upload a .wav file to begin.", lang))

