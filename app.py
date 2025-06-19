import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64

# --- 다국어 딕셔너리 ---
lang_dict = {
    "en": {
        "title": "Music Genre Classifier",
        "upload": "Upload one or more .wav files",
        "select_model": "Choose a model",
        "download_rf": "⬇️ Download Random Forest Classification Report",
        "download_svm": "⬇️ Download SVM Classification Report",
        "predicted_genre": "Predicted Genre",
        "show_heatmap": "Show MFCC Heatmap",
        "accuracy_summary": "Model Accuracy Summary",
        "accuracy_rf": "Random Forest Accuracy",
        "accuracy_svm": "SVM Accuracy",
        "best_genres": "Best performing genres",
        "about_app": "About This App",
        "model_performance": "Model Performance Metrics",
        "select_file": "Select a file to classify",
        "choose_language": "Choose Language / 언어 선택",
        "start_info": "Please upload one or more .wav files to get started."
    },
    "ko": {
        "title": "음악 장르 분류기",
        "upload": ".wav 파일을 업로드하세요",
        "select_model": "모델 선택",
        "download_rf": "⬇️ 랜덤 포레스트 분류 리포트 다운로드",
        "download_svm": "⬇️ SVM 분류 리포트 다운로드",
        "predicted_genre": "예측된 장르",
        "show_heatmap": "MFCC 히트맵 보기",
        "accuracy_summary": "모델 정확도 요약",
        "accuracy_rf": "랜덤 포레스트 정확도",
        "accuracy_svm": "SVM 정확도",
        "best_genres": "성능이 좋은 장르",
        "about_app": "앱 정보",
        "model_performance": "모델 성능 지표",
        "select_file": "분류할 파일 선택",
        "choose_language": "언어 선택 / Choose Language",
        "start_info": "하나 이상의 .wav 파일을 업로드 해주세요."
    },
    # 추가 언어들도 같은 형식으로 추가하세요
}

# --- 언어 선택 UI ---
language = st.sidebar.selectbox(
    "Choose Language / 언어 선택",
    options=list(lang_dict.keys()),
    index=0
)
texts = lang_dict[language]

# 페이지 설정
st.set_page_config(page_title=texts["title"], layout="centered")

# 모델 선택
model_option = st.radio(texts["select_model"], ("Random Forest", "SVM"))
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
st.markdown(f"""
<h1 style='text-align: center; color: #FF4B4B;'>🎵 {texts['title']}</h1>
<p style='text-align: center;'>{texts['upload']}</p>
<hr>
""", unsafe_allow_html=True)

# 샘플 오디오 다운로드 버튼 (사이드바에 추가)
with open("sample.wav", "rb") as audio_file:
    rf_data = open(rf_report_path, "rb").read()
    svm_data = open(svm_report_path, "rb").read()
    rf_b64 = base64.b64encode(rf_data).decode('utf-8')
    svm_b64 = base64.b64encode(svm_data).decode('utf-8')

    st.sidebar.header(texts["about_app"])

    st.sidebar.markdown(f"""
    **Created by Suhwa Seong**  
    Model: {model_option}  
    Features: 13 MFCCs (mean + std)  
    Accuracy: ~64% if Random Forest else ~61%
    """)

    st.sidebar.download_button(
        label=texts["download_rf"],
        data=rf_data,
        file_name="rf_classification_report.csv",
        mime="text/csv",
    )
    st.sidebar.download_button(
        label=texts["download_svm"],
        data=svm_data,
        file_name="svm_classification_report.csv",
        mime="text/csv",
    )
    st.sidebar.download_button(
        label="⬇️ Download Sample Audio (.wav)",
        data=audio_file,
        file_name="sample.wav",
        mime="audio/wav"
    )

    st.sidebar.header(texts["model_performance"])
    st.sidebar.dataframe(report_metrics)
    st.sidebar.bar_chart(report_metrics)

# 여러 파일 업로드
uploaded_files = st.file_uploader(texts["upload"], type=["wav"], accept_multiple_files=True)

if uploaded_files:
    filenames = [file.name for file in uploaded_files]
    selected_file = st.selectbox(texts["select_file"], filenames)

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
        st.success(f"🎶 **{texts['predicted_genre']}:** `{prediction[0].capitalize()}`")

        # 예측 확률 보기 (가능할 경우)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            classes_encoded = model.classes_
            classes = label_encoder.inverse_transform(classes_encoded)
            proba_dict = dict(zip(classes, proba))
            st.markdown("### 🔍 Prediction Probabilities")
            st.bar_chart(proba_dict)

        # 정확도 요약
        with st.expander(texts["accuracy_summary"]):
            st.markdown(f"""
            - **{texts['accuracy_rf']}:** ~64%  
            - **{texts['accuracy_svm']}:** ~61%  
            - {texts['best_genres']}: 🎼 `Classical`, 🤘 `Metal`, 🎷 `Jazz`
            """)

        # MFCC 히트맵 시각화
        if st.checkbox(texts["show_heatmap"]):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
            ax.set_title("MFCC Features")
            ax.set_xlabel("Time")
            ax.set_ylabel("MFCC Coefficients")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong while processing the audio file.\n\nError: {e}")
else:
    st.info(texts["start_info"])
