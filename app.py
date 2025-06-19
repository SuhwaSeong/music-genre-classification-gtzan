import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf

# 🎨 페이지 설정
st.set_page_config(page_title="Music Genre Classifier", layout="centered")

# 🎯 모델 선택
model_option = st.radio("Choose a model", ("Random Forest", "SVM"))
model_file = "model.pkl" if model_option == "Random Forest" else "svm_model.pkl"
model = joblib.load(model_file)

# 🎵 앱 헤더
st.markdown("""
<h1 style='text-align: center; color: #FF4B4B;'>🎵 Music Genre Classifier</h1>
<p style='text-align: center;'>Upload a <b>.wav</b> file and I'll try to guess the genre using machine learning!</p>
<hr>
""", unsafe_allow_html=True)

# ⬇️ 샘플 오디오 다운로드 버튼
with open("sample.wav", "rb") as audio_file:
    st.download_button(
        label="⬇️ Download Sample Audio (.wav)",
        data=audio_file,
        file_name="sample.wav",
        mime="audio/wav"
    )

# 📌 사이드바 정보
st.sidebar.header("📌 About This App")
if model_option == "Random Forest":
    st.sidebar.markdown("""
    **Created by Suhwa Seong**  
    Model: Random Forest  
    Features: 13 MFCCs (mean + std)  
    Accuracy: ~64%
    """)
else:
    st.sidebar.markdown("""
    **Created by Suhwa Seong**  
    Model: Support Vector Machine (SVM)  
    Features: 13 MFCCs (mean + std)  
    Accuracy: ~61%
    """)

# 📁 오디오 업로드
uploaded_file = st.file_uploader("🎵 Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        # 🎧 오디오 재생
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        uploaded_file.seek(0)  # 파일 포인터 초기화

        # 🎼 MFCC 특징 추출
        y, sr = librosa.load(uploaded_file, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features = np.concatenate((mfcc_mean, mfcc_std)).reshape(1, -1)

        # 🎯 예측 수행
        prediction = model.predict(features)
        st.success(f"🎶 **Predicted Genre:** `{prediction[0].capitalize()}`")

        # 📈 예측 확률 보기 (Random Forest만 지원)
        if model_option == "Random Forest" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            classes = model.classes_
            proba_dict = dict(zip(classes, proba))
            st.markdown("### 🔍 Prediction Probabilities")
            st.bar_chart(proba_dict)

        # 📊 정확도 요약
        with st.expander("📊 Model Accuracy Summary"):
            st.markdown("""
            - **Random Forest Accuracy:** ~64%  
            - **SVM Accuracy:** ~61%  
            - Best performing genres: 🎼 `Classical`, 🤘 `Metal`, 🎷 `Jazz`
            """)

        # 🌡 MFCC 히트맵 시각화
        if st.checkbox("Show MFCC Heatmap"):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
            ax.set_title("MFCC Features")
            ax.set_xlabel("Time")
            ax.set_ylabel("MFCC Coefficients")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong while processing the audio file.\n\nError: {e}")
else:
    st.info("Please upload a .wav file to get started.")
