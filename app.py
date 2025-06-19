import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit 페이지 설정
st.set_page_config(page_title="Music Genre Classifier", layout="centered")

# 🔍 모델 선택
model_option = st.radio("Choose a model", ("Random Forest", "SVM"))
model_file = "model.pkl" if model_option == "Random Forest" else "svm_model.pkl"
model = joblib.load(model_file)

# 🔢 정확도 정보
accuracy_info = {
    "Random Forest": "64%",
    "SVM": "61%"
}

# 🔻 상단 UI 헤더
st.markdown("""
<h1 style='text-align: center; color: #FF4B4B;'>🎧 Music Genre Classifier</h1>
<p style='text-align: center;'>Upload a <b>.wav</b> file and I'll try to guess the genre using machine learning!</p>
<hr>
""", unsafe_allow_html=True)

# 📌 사이드바 정보
st.sidebar.header("📌 About This App")
st.sidebar.markdown("""
**Created by Suhwa Seong**  
Model: Random Forest / SVM  
Features: 13 MFCCs (mean + std)

### ✅ Model Accuracy
**Random Forest:** 64%  
**SVM:** 61%
""")

# 🎵 파일 업로드
uploaded_file = st.file_uploader("🎵 Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        y, sr = librosa.load(uploaded_file, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features = np.concatenate((mfcc_mean, mfcc_std)).reshape(1, -1)

        prediction = model.predict(features)

        st.success(f"🎶 **Predicted Genre:** `{prediction[0].capitalize()}`")

        if st.checkbox("Show MFCC Heatmap"):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
            plt.title("MFCC Features")
            plt.xlabel("Time")
            plt.ylabel("MFCC Coefficients")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong while processing the audio file.\n\nError: {e}")
else:
    st.info("Please upload a .wav file to get started.")
