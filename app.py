import streamlit as st
from googletrans import Translator
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import gdown
from io import BytesIO
import tensorflow as tf
import pickle
from sklearn.metrics import confusion_matrix
import datetime

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

# 테마 스타일 선택
style = st.sidebar.selectbox("🎨 Select Plot Style", plt.style.available)
plt.style.use(style)

# 초기 안내
st.sidebar.info(translate_text("Please select a language and upload a .wav file.", lang))

# 섹션: 정확도 비교
st.markdown("## 📊 Model Accuracy Comparison")

# 정확도 파일 로드
try:
    acc_rf = pd.read_csv("rf_classification_report.csv", index_col=0).loc["accuracy"].values[0]
    acc_svm = pd.read_csv("svm_classification_report.csv", index_col=0).loc["accuracy"].values[0]
    acc_cnn = pd.read_csv("cnn_classification_report.csv", index_col=0).loc["accuracy"].values[0]
except Exception as e:
    st.error(f"⚠️ 정확도 파일을 불러오는 데 실패했습니다: {e}")
    st.stop()

# 정확도 그래프
df_acc = pd.DataFrame({"Model": ["Random Forest", "SVM", "CNN"], "Accuracy": [acc_rf, acc_svm, acc_cnn]})
fig, ax = plt.subplots()
bars = ax.bar(df_acc["Model"], df_acc["Accuracy"], color=["orange", "lightgreen", "skyblue"])
ax.set_ylim(0, 1)
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2%}", ha='center')
st.pyplot(fig)

# 파일 다운로드 함수
def download_file_if_missing(file_name, file_id):
    if not os.path.exists(file_name):
        with st.spinner(f"📃 Downloading {file_name}..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)

# 필요한 파일 다운로드
files_to_download = {
    "rf_model.pkl": "1oBV5HpsvgoCLr5CYLvrmR6wbMiNP89Gi",
    "svm_model.pkl": "1B3ftW3aIze7gC_QrDK7WAqROBs19jwHt",
    "scaler.pkl": "1tbkqFV95yHrvsLd9NpUvj1QSRpIoen0k",
    "label_encoder.pkl": "1i3wvy68pVMpzjK5y2ny3OeB5KQGEGcQs",
    "rf_classification_report.csv": "1WEkLBZrsFcdFoLLeGH737Feqf5ihmZsB",
    "svm_classification_report.csv": "1FmegZMchjzuX0Tr6aF7rxrvlbmp_Ei-d"
}
for f, fid in files_to_download.items():
    download_file_if_missing(f, fid)

@st.cache_resource
def load_cnn_model():
    cnn_model_path = "cnn_genre_model.h5"
    if not os.path.exists(cnn_model_path):
        with st.spinner("📥 Downloading CNN model (.h5) from Google Drive..."):
            gdown.download("https://drive.google.com/uc?id=1ZvB83SkO3bY7q3TXd9A5th25MnRIGlYF", cnn_model_path, quiet=False)
    model = tf.keras.models.load_model(cnn_model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_files(name):
    model = joblib.load(f"{name.lower()}_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

def extract_features(audio_bytes, n_mfcc=13):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).reshape(1, -1), mfcc, y, sr

def extract_mel_spectrogram(audio_bytes, max_len=128):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    pad_width = max(0, max_len - mel_db.shape[1])
    mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant') if pad_width > 0 else mel_db[:, :max_len]
    return mel_db[np.newaxis, ..., np.newaxis], mel_db, y, sr

genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

st.title(translate_text("🎵 Music Genre Classifier (CNN included)", lang))
model_option = st.selectbox(translate_text("Select model", lang), ["Random Forest", "SVM", "CNN"])
uploaded_file = st.file_uploader(translate_text("Upload a .wav file", lang), type=["wav"])

if uploaded_file:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

    prediction_result = {}

    if model_option == "CNN":
        model = load_cnn_model()
        features, mel, y, sr = extract_mel_spectrogram(audio_bytes)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_label = genre_labels[predicted_index]
        prediction_result = dict(zip(genre_labels, prediction[0]))

        st.success(f"{translate_text('Predicted genre:', lang)} `{predicted_label}`")
        st.markdown(translate_text("### 🔍 Prediction Probabilities", lang))
        st.bar_chart(prediction_result)

        # 🎶 오디오 파형
        if st.checkbox("Show Waveform"):
            fig, ax = plt.subplots()
            ax.plot(y)
            ax.set_title("Waveform")
            st.pyplot(fig)
            plt.close(fig)

        if st.checkbox("Show Mel Spectrogram"):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mel, cmap="YlGnBu", ax=ax)
            ax.set(title="Mel Spectrogram", xlabel="Time", ylabel="Mel Bands")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### 🧩 CNN Confusion Matrix")
        try:
            y_test = pd.read_csv("cnn_y_test.csv").values.flatten()
            y_pred = pd.read_csv("cnn_y_pred.csv").values.flatten()
            with open("label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)

            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu",
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix - CNN")
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        except Exception as e:
            st.warning("⚠️ Failed to load confusion matrix. Please ensure CNN test data is available.")
            st.exception(e)

    else:
        model, scaler, label_encoder = load_model_files(model_option)
        features, mfcc, y, sr = extract_features(audio_bytes)
        features_scaled = scaler.transform(features)
        pred_encoded = model.predict(features_scaled)
        pred_label = label_encoder.inverse_transform(pred_encoded)
        st.success(f"{translate_text('Predicted genre:', lang)} `{pred_label[0]}`")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            class_labels = label_encoder.classes_
            prediction_result = dict(zip(class_labels, proba))
            st.markdown(translate_text("### 🔍 Prediction Probabilities", lang))
            st.bar_chart(prediction_result)

        if st.checkbox("Show MFCC Heatmap"):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
            ax.set(title="MFCC Features", xlabel="Time", ylabel="MFCC Coefficients")
            st.pyplot(fig)
            plt.close(fig)

        if st.checkbox("Show Waveform"):
            fig, ax = plt.subplots()
            ax.plot(y)
            ax.set_title("Waveform")
            st.pyplot(fig)
            plt.close(fig)

    # 📥 예측 결과 CSV 저장
    result_df = pd.DataFrame.from_dict(prediction_result, orient='index', columns=['Probability'])
    result_df.index.name = 'Genre'
    csv = result_df.to_csv().encode('utf-8')
    st.download_button(label="📥 Download Prediction Result (CSV)", data=csv, file_name="prediction_result.csv", mime="text/csv")

    # 🌍 사용자 로그 저장
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        "datetime": now,
        "model": model_option,
        "predicted_label": predicted_label if model_option == "CNN" else pred_label[0],
        "filename": uploaded_file.name
    }
    with open("user_logs.csv", "a") as f:
        f.write(f"{log_data['datetime']},{log_data['filename']},{log_data['model']},{log_data['predicted_label']}\n")

else:
    st.info(translate_text("📂 Please upload a .wav file to begin.", lang))
