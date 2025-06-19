import streamlit as st
import librosa
import numpy as np
import joblib

# 해당 .pkl 파일을 미리 저장해두었다고 값
model = joblib.load("model.pkl")

# 해당 복수문의 만약 예측 표시에 이용할 MFCC 시각화용 figure
import matplotlib.pyplot as plt
import seaborn as sns

# 하단 화면 UI 개정 초기화
st.set_page_config(page_title="Music Genre Classifier", layout="centered")

# 바위 화면 헤더
st.markdown("""
<h1 style='text-align: center; color: #FF4B4B;'>🎧 Music Genre Classifier</h1>
<p style='text-align: center;'>Upload a <b>.wav</b> file and I'll try to guess the genre using machine learning!</p>
<hr>
""", unsafe_allow_html=True)

# 사이드바 정보
st.sidebar.header("📌 About This App")
st.sidebar.markdown("""
**Created by Suhwa Seong**  
Model: Random Forest  
Features: 13 MFCCs (mean + std)
""")

# 파일 업로드
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

        # 히트맵 체크박스로 선택 시
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
