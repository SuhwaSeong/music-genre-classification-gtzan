import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # 추가

# 페이지 설정
st.set_page_config(page_title="Music Genre Classifier", layout="centered")

# 모델 선택
model_option = st.radio("Choose a model", ("Random Forest", "SVM"))
model_file = "model.pkl" if model_option == "Random Forest" else "svm_model.pkl"
model = joblib.load(model_file)
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 평가 리포트 CSV 경로
rf_report_path = "rf_classification_report.csv"
svm_report_path = "svm_classification_report.csv"

# 선택한 모델에 따라 평가 리포트 불러오기
if model_option == "Random Forest":
    report_df = pd.read_csv(rf_report_path, index_col=0)
else:
    report_df = pd.read_csv(svm_report_path, index_col=0)

# 주요 지표만 선택
metrics = ["precision", "recall", "f1-score"]
report_metrics = report_df.loc[:, metrics]

# 앱 헤더
st.markdown("""
<h1 style='text-align: center; color: #FF4B4B;'>🎵 Music Genre Classifier</h1>
<p style='text-align: center;'>Upload one or more <b>.wav</b> files and select which one to classify!</p>
<hr>
""", unsafe_allow_html=True)

# 샘플 오디오 다운로드 버튼
with open("sample.wav", "rb") as audio_file:
    st.download_button(
        label="⬇️ Download Sample Audio (.wav)",
        data=audio_file,
        file_name="sample.wav",
        mime="audio/wav"
    )

# 사이드바 정보 및 성능 리포트 시각화
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

st.sidebar.header("📊 Model Performance Metrics")
st.sidebar.dataframe(report_metrics)
st.sidebar.bar_chart(report_metrics)

# 여러 파일 업로드
uploaded_files = st.file_uploader("🎵 Choose WAV files", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    filenames = [file.name for file in uploaded_files]
    selected_file = st.selectbox("Select a file to classify", filenames)

    # 선택한 파일 객체 찾기
    file_obj = next(file for file in uploaded_files if file.name == selected_file)

    try:
        # 오디오 재생
        audio_bytes = file_obj.read()
        st.audio(audio_bytes, format='audio/wav')
        file_obj.seek(0)  # 파일 포인터 초기화

        # MFCC 특징 추출
        y, sr = librosa.load(file_obj, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features = np.concatenate((mfcc_mean, mfcc_std)).reshape(1, -1)

        # 스케일링
        features_scaled = scaler.transform(features)

        # 예측 수행
        prediction_encoded = model.predict(features_scaled)
        prediction = label_encoder.inverse_transform(prediction_encoded)
        st.success(f"🎶 **Predicted Genre:** `{prediction[0].capitalize()}`")

        # 예측 확률 보기 (가능할 경우)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            classes_encoded = model.classes_
            classes = label_encoder.inverse_transform(classes_encoded)
            proba_dict = dict(zip(classes, proba))
            st.markdown("### 🔍 Prediction Probabilities")
            st.bar_chart(proba_dict)

        # 정확도 요약
        with st.expander("📊 Model Accuracy Summary"):
            st.markdown("""
            - **Random Forest Accuracy:** ~64%  
            - **SVM Accuracy:** ~61%  
            - Best performing genres: 🎼 `Classical`, 🤘 `Metal`, 🎷 `Jazz`
            """)

        # MFCC 히트맵 시각화
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
    st.info("Please upload one or more .wav files to get started.")



