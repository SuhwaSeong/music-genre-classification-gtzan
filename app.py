import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
import tempfile  # 임시 파일 처리를 위한 모듈 (For temporary file handling)
import soundfile as sf  # 오디오 파일 읽고 쓰기용 (For reading/writing audio files)
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# --- 다국어 딕셔너리 (Languages dictionary) ---
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
        "start_info": "Please upload one or more .wav files to get started.",
        "mic_start_info": "Click the button above to start recording."  # Mic recording start prompt
    },
    "de": {
        "title": "Musikgenre-Klassifikator",
        "upload": "Laden Sie eine oder mehrere .wav-Dateien hoch",
        "select_model": "Wählen Sie ein Modell",
        "download_rf": "⬇️ Random Forest Klassifikationsbericht herunterladen",
        "download_svm": "⬇️ SVM Klassifikationsbericht herunterladen",
        "predicted_genre": "Vorhergesagtes Genre",
        "show_heatmap": "MFCC Heatmap anzeigen",
        "accuracy_summary": "Modellgenauigkeitszusammenfassung",
        "accuracy_rf": "Random Forest Genauigkeit",
        "accuracy_svm": "SVM Genauigkeit",
        "best_genres": "Beste Genres",
        "about_app": "Über diese App",
        "model_performance": "Modellleistungsmetriken",
        "select_file": "Wählen Sie eine Datei zur Klassifizierung",
        "choose_language": "Sprache auswählen / Choose Language / 언어 선택",
        "start_info": "Bitte laden Sie eine oder mehrere .wav-Dateien hoch, um zu beginnen.",
        "mic_start_info": "Klicken Sie oben auf die Schaltfläche, um die Aufnahme zu starten."  # Mic recording start prompt
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
        "start_info": "하나 이상의 .wav 파일을 업로드 해주세요.",
        "mic_start_info": "녹음을 시작하려면 위 버튼을 클릭하세요."  # Mic recording start prompt
    },
"pl": {
    "title": "Klasyfikator gatunków muzycznych",
    "upload": "Prześlij jeden lub więcej plików .wav",
    "select_model": "Wybierz model",
    "download_rf": "⬇️ Pobierz raport klasyfikacji Random Forest",
    "download_svm": "⬇️ Pobierz raport klasyfikacji SVM",
    "predicted_genre": "Przewidywany gatunek",
    "show_heatmap": "Pokaż mapę ciepła MFCC",
    "accuracy_summary": "Podsumowanie dokładności modelu",
    "accuracy_rf": "Dokładność Random Forest",
    "accuracy_svm": "Dokładność SVM",
    "best_genres": "Najlepsze gatunki",
    "about_app": "O aplikacji",
    "model_performance": "Metryki wydajności modelu",
    "select_file": "Wybierz plik do klasyfikacji",
    "choose_language": "Wybierz język / Choose Language / 언어 선택",
    "start_info": "Proszę przesłać jeden lub więcej plików .wav, aby rozpocząć.",
    "mic_start_info": "Kliknij przycisk powyżej, aby rozpocząć nagrywanie."  # 녹음 시작 안내 (Mic recording start info)
},
"hi": {
    "title": "संगीत शैली वर्गीकर्ता",
    "upload": ".wav फ़ाइल अपलोड करें",
    "select_model": "मॉडल चुनें",
    "download_rf": "⬇️ रैंडम फॉरेस्ट वर्गीकरण रिपोर्ट डाउनलोड करें",
    "download_svm": "⬇️ एसवीएम वर्गीकरण रिपोर्ट डाउनलोड करें",
    "predicted_genre": "अनुमानित शैली",
    "show_heatmap": "MFCC हीटमैप दिखाएँ",
    "accuracy_summary": "मॉडल सटीकता सारांश",
    "accuracy_rf": "रैंडम फॉरेस्ट सटीकता",
    "accuracy_svm": "एसवीएम सटीकता",
    "best_genres": "सर्वश्रेष्ठ प्रदर्शन वाले शैलियाँ",
    "about_app": "इस ऐप के बारे में",
    "model_performance": "मॉडल प्रदर्शन मेट्रिक्स",
    "select_file": "वर्गीकृत करने के लिए फ़ाइल चुनें",
    "choose_language": "भाषा चुनें / Choose Language / 언어 선택",
    "start_info": "शुरू करने के लिए एक या अधिक .wav फ़ाइलें अपलोड करें।",
    "mic_start_info": "रिकॉर्डिंग शुरू करने के लिए ऊपर दिए गए बटन पर क्लिक करें।"
},
"ta": {
    "title": "பாடல் வகை வகைப்பான்",
    "upload": ".wav கோப்புகளை பதிவேற்றவும்",
    "select_model": "மாதிரியைத் தேர்ந்தெடுக்கவும்",
    "download_rf": "⬇️ ரேண்டம் ஃபாரெஸ்ட் வகைப்பாட்டு அறிக்கை பதிவிறக்கு",
    "download_svm": "⬇️ எஸ்விஎம் வகைப்பாட்டு அறிக்கை பதிவிறக்கு",
    "predicted_genre": "முன்னறிவிப்பு வகை",
    "show_heatmap": "MFCC ஹீட்மாப் காண்க",
    "accuracy_summary": "மாதிரி துல்லியத் தொகுப்பு",
    "accuracy_rf": "ரேண்டம் ஃபாரெஸ்ட் துல்லியம்",
    "accuracy_svm": "எஸ்விஎம் துல்லியம்",
    "best_genres": "சிறந்த செயல்திறன் வகைகள்",
    "about_app": "இந்த செயலியின் பற்றி",
    "model_performance": "மாதிரி செயல்திறன் அளவுகோல்கள்",
    "select_file": "வகைப்படுத்த கோப்பைத் தேர்ந்தெடு",
    "choose_language": "மொழி தேர்ந்தெடு / Choose Language / 언어 선택",
    "start_info": "தொடங்க ஒரு அல்லது அதற்கு மேற்பட்ட .wav கோப்புகளை பதிவேற்றவும்.",
    "mic_start_info": "பதிவு செய்ய ஆரம்பிக்க மேலுள்ள பொத்தானை அழுத்தவும்."
},
"zh": {
    "title": "音乐类别分类器",
    "upload": "上传一个或多个.wav文件",
    "select_model": "选择模型",
    "download_rf": "⬇️ 下载随机森林分类报告",
    "download_svm": "⬇️ 下载SVM分类报告",
    "predicted_genre": "预测的类别",
    "show_heatmap": "显示MFCC热图",
    "accuracy_summary": "模型准确度摘要",
    "accuracy_rf": "随机森林准确度",
    "accuracy_svm": "SVM准确度",
    "best_genres": "表现最佳的类别",
    "about_app": "关于此应用",
    "model_performance": "模型性能指标",
    "select_file": "选择要分类的文件",
    "choose_language": "选择语言 / Choose Language / 언어 선택",
    "start_info": "请上传一个或多个.wav文件开始使用。",
    "mic_start_info": "点击上方按钮开始录音。"
},
"hk": {
    "title": "音樂類型分類器",
    "upload": "上載一個或多個.wav檔案",
    "select_model": "選擇模型",
    "download_rf": "⬇️ 下載隨機森林分類報告",
    "download_svm": "⬇️ 下載SVM分類報告",
    "predicted_genre": "預測類別",
    "show_heatmap": "顯示MFCC熱圖",
    "accuracy_summary": "模型準確率摘要",
    "accuracy_rf": "隨機森林準確率",
    "accuracy_svm": "SVM準確率",
    "best_genres": "表現最佳類別",
    "about_app": "關於此應用程式",
    "model_performance": "模型性能指標",
    "select_file": "選擇要分類的檔案",
    "choose_language": "選擇語言 / Choose Language / 언어 선택",
    "start_info": "請上載一個或多個.wav檔案開始使用。",
    "mic_start_info": "點擊上方按鈕開始錄音。"
},
"ja": {
    "title": "音楽ジャンル分類器",
    "upload": "1つ以上の.wavファイルをアップロードしてください",
    "select_model": "モデルを選択",
    "download_rf": "⬇️ ランダムフォレスト分類レポートをダウンロード",
    "download_svm": "⬇️ SVM分類レポートをダウンロード",
    "predicted_genre": "予測されたジャンル",
    "show_heatmap": "MFCCヒートマップを表示",
    "accuracy_summary": "モデルの精度概要",
    "accuracy_rf": "ランダムフォレストの精度",
    "accuracy_svm": "SVMの精度",
    "best_genres": "最も性能が良いジャンル",
    "about_app": "このアプリについて",
    "model_performance": "モデルパフォーマンス指標",
    "select_file": "分類するファイルを選択",
    "choose_language": "言語を選択 / Choose Language / 언어 선택",
    "start_info": "開始するには1つ以上の.wavファイルをアップロードしてください。",
    "mic_start_info": "録音を開始するには上のボタンをクリックしてください。"
},
"fr": {
    "title": "Classificateur de genre musical",
    "upload": "Téléchargez un ou plusieurs fichiers .wav",
    "select_model": "Choisir un modèle",
    "download_rf": "⬇️ Télécharger le rapport de classification Random Forest",
    "download_svm": "⬇️ Télécharger le rapport de classification SVM",
    "predicted_genre": "Genre prédit",
    "show_heatmap": "Afficher la carte thermique MFCC",
    "accuracy_summary": "Résumé de la précision du modèle",
    "accuracy_rf": "Précision Random Forest",
    "accuracy_svm": "Précision SVM",
    "best_genres": "Genres les mieux performants",
    "about_app": "À propos de cette application",
    "model_performance": "Mesures de performance du modèle",
    "select_file": "Sélectionnez un fichier à classer",
    "choose_language": "Choisir la langue / Choose Language / 언어 선택",
    "start_info": "Veuillez télécharger un ou plusieurs fichiers .wav pour commencer.",
    "mic_start_info": "Cliquez sur le bouton ci-dessus pour commencer l'enregistrement."
},
"it": {
    "title": "Classificatore di genere musicale",
    "upload": "Carica uno o più file .wav",
    "select_model": "Scegli un modello",
    "download_rf": "⬇️ Scarica il rapporto di classificazione Random Forest",
    "download_svm": "⬇️ Scarica il rapporto di classificazione SVM",
    "predicted_genre": "Genere previsto",
    "show_heatmap": "Mostra la mappa di calore MFCC",
    "accuracy_summary": "Riepilogo accuratezza modello",
    "accuracy_rf": "Accuratezza Random Forest",
    "accuracy_svm": "Accuratezza SVM",
    "best_genres": "Generi con migliori prestazioni",
    "about_app": "Informazioni su questa app",
    "model_performance": "Metriche di prestazione del modello",
    "select_file": "Seleziona un file da classificare",
    "choose_language": "Scegli la lingua / Choose Language / 언어 선택",
    "start_info": "Carica uno o più file .wav per iniziare.",
    "mic_start_info": "Fai clic sul pulsante sopra per iniziare la registrazione."
},
"ru": {
    "title": "Классификатор музыкальных жанров",
    "upload": "Загрузите один или несколько файлов .wav",
    "select_model": "Выберите модель",
    "download_rf": "⬇️ Скачать отчет классификации Random Forest",
    "download_svm": "⬇️ Скачать отчет классификации SVM",
    "predicted_genre": "Предсказанный жанр",
    "show_heatmap": "Показать тепловую карту MFCC",
    "accuracy_summary": "Обзор точности модели",
    "accuracy_rf": "Точность Random Forest",
    "accuracy_svm": "Точность SVM",
    "best_genres": "Лучшие жанры",
    "about_app": "Об этом приложении",
    "model_performance": "Метрики производительности модели",
    "select_file": "Выберите файл для классификации",
    "choose_language": "Выберите язык / Choose Language / 언어 선택",
    "start_info": "Пожалуйста, загрузите один или несколько файлов .wav для начала.",
    "mic_start_info": "Нажмите кнопку выше, чтобы начать запись."
},
"es": {
    "title": "Clasificador de Géneros Musicales",
    "upload": "Sube uno o más archivos .wav",
    "select_model": "Elige un modelo",
    "download_rf": "⬇️ Descargar informe de clasificación Random Forest",
    "download_svm": "⬇️ Descargar informe de clasificación SVM",
    "predicted_genre": "Género Predicho",
    "show_heatmap": "Mostrar mapa de calor MFCC",
    "accuracy_summary": "Resumen de precisión del modelo",
    "accuracy_rf": "Precisión Random Forest",
    "accuracy_svm": "Precisión SVM",
    "best_genres": "Géneros con mejor desempeño",
    "about_app": "Acerca de esta aplicación",
    "model_performance": "Métricas de desempeño del modelo",
    "select_file": "Selecciona un archivo para clasificar",
    "choose_language": "Elige idioma / Choose Language / 언어 선택",
    "start_info": "Por favor, sube uno o más archivos .wav para comenzar.",
    "mic_start_info": "Haga clic en el botón de arriba para comenzar la grabación."
},
"ar": {
    "title": "مصنف نوع الموسيقى",
    "upload": "قم بتحميل ملف أو أكثر بصيغة .wav",
    "select_model": "اختر نموذجًا",
    "download_rf": "⬇️ تحميل تقرير تصنيف Random Forest",
    "download_svm": "⬇️ تحميل تقرير تصنيف SVM",
    "predicted_genre": "النوع المتوقع",
    "show_heatmap": "عرض خريطة الحرارة MFCC",
    "accuracy_summary": "ملخص دقة النموذج",
    "accuracy_rf": "دقة Random Forest",
    "accuracy_svm": "دقة SVM",
    "best_genres": "أفضل الأنواع أداءً",
    "about_app": "حول هذا التطبيق",
    "model_performance": "مقاييس أداء النموذج",
    "select_file": "اختر ملفًا للتصنيف",
    "choose_language": "اختر اللغة / Choose Language / 언어 선택",
    "start_info": "يرجى تحميل ملف واحد أو أكثر بصيغة .wav للبدء.",
    "mic_start_info": "انقر فوق الزر أعلاه لبدء التسجيل."
},
"pt": {
    "title": "Classificador de Gêneros Musicais",
    "upload": "Faça upload de um ou mais arquivos .wav",
    "select_model": "Escolha um modelo",
    "download_rf": "⬇️ Baixar relatório de classificação Random Forest",
    "download_svm": "⬇️ Baixar relatório de classificação SVM",
    "predicted_genre": "Gênero Previsto",
    "show_heatmap": "Mostrar mapa de calor MFCC",
    "accuracy_summary": "Resumo de precisão do modelo",
    "accuracy_rf": "Precisão Random Forest",
    "accuracy_svm": "Precisão SVM",
    "best_genres": "Melhores gêneros",
    "about_app": "Sobre este aplicativo",
    "model_performance": "Métricas de desempenho do modelo",
    "select_file": "Selecione um arquivo para classificar",
    "choose_language": "Escolha o idioma / Choose Language / 언어 선택",
    "start_info": "Por favor, faça upload de um ou mais arquivos .wav para começar.",
    "mic_start_info": "Clique no botão acima para começar a gravação."
},
"vi": {
    "title": "Bộ Phân Loại Thể Loại Nhạc",
    "upload": "Tải lên một hoặc nhiều file .wav",
    "select_model": "Chọn mô hình",
    "download_rf": "⬇️ Tải xuống báo cáo phân loại Random Forest",
    "download_svm": "⬇️ Tải xuống báo cáo phân loại SVM",
    "predicted_genre": "Thể loại dự đoán",
    "show_heatmap": "Hiển thị bản đồ nhiệt MFCC",
    "accuracy_summary": "Tóm tắt độ chính xác mô hình",
    "accuracy_rf": "Độ chính xác Random Forest",
    "accuracy_svm": "Độ chính xác SVM",
    "best_genres": "Thể loại hoạt động tốt nhất",
    "about_app": "Về ứng dụng này",
    "model_performance": "Chỉ số hiệu suất mô hình",
    "select_file": "Chọn tệp để phân loại",
    "choose_language": "Chọn ngôn ngữ / Choose Language / 언어 선택",
    "start_info": "Vui lòng tải lên một hoặc nhiều tệp .wav để bắt đầu.",
    "mic_start_info": "Nhấp vào nút ở trên để bắt đầu ghi âm."
},
"tr": {
    "title": "Müzik Türü Sınıflandırıcı",
    "upload": "Bir veya daha fazla .wav dosyası yükleyin",
    "select_model": "Bir model seçin",
    "download_rf": "⬇️ Random Forest Sınıflandırma Raporunu İndir",
    "download_svm": "⬇️ SVM Sınıflandırma Raporunu İndir",
    "predicted_genre": "Tahmin Edilen Tür",
    "show_heatmap": "MFCC Isı Haritasını",
    "mic_start_info": "Kayda başlamak için yukarıdaki düğmeye tıklayın."
    },  # 각 언어별 딕셔너리 닫힘
}  # lang_dict 전체 닫는 중괄호

# --- 언어 선택 UI (Language selection UI) ---
language = st.sidebar.selectbox(
    "Choose Language / 언어 선택",
    options=list(lang_dict.keys()),
    index=0
)
texts = lang_dict[language]

# 페이지 설정 (Page setup)
st.set_page_config(page_title=texts["title"], layout="centered")

# 모델 선택 (Model selection)
model_option = st.radio(texts["select_model"], ("Random Forest", "SVM"))
model_file = "model.pkl" if model_option == "Random Forest" else "svm_model.pkl"
model = joblib.load(model_file)  # 모델 불러오기 (Load model)
scaler = joblib.load("scaler.pkl")  # 스케일러 불러오기 (Load scaler)
label_encoder = joblib.load("label_encoder.pkl")  # 레이블 인코더 불러오기 (Load label encoder)

# 평가 리포트 CSV 경로 (CSV paths for evaluation reports)
rf_report_path = "rf_classification_report.csv"
svm_report_path = "svm_classification_report.csv"

# 선택한 모델에 따라 평가 리포트 불러오기 (Load evaluation report based on model)
if model_option == "Random Forest":
    with open(rf_report_path, "rb") as rf_file:
        rf_data = rf_file.read()
    report_df = pd.read_csv(rf_report_path, index_col=0)
else:
    with open(svm_report_path, "rb") as svm_file:
        svm_data = svm_file.read()
    report_df = pd.read_csv(svm_report_path, index_col=0)

# 주요 지표만 선택 (Select main metrics)
metrics = ["precision", "recall", "f1-score"]
report_metrics = report_df.loc[:, metrics]

# 앱 헤더 (App header)
st.markdown(f"""
<h1 style='text-align: center; color: #FF4B4B;'>🎵 {texts['title']}</h1>
<p style='text-align: center;'>{texts['upload']}</p>
<hr>
""", unsafe_allow_html=True)

# 샘플 오디오 다운로드 버튼 (Sample audio download buttons)
with open("sample.wav", "rb") as audio_file:
    st.sidebar.header(texts["about_app"])

    st.sidebar.markdown(f"""
    **Created by Suhwa Seong**  
    Model: {model_option}  
    Features: 13 MFCCs (mean + std)  
    Accuracy: ~64% if Random Forest else ~61%
    """)

    # 리포트 다운로드 버튼 (Report download buttons)
    if model_option == "Random Forest":
        st.sidebar.download_button(
            label=texts["download_rf"],
            data=rf_data,
            file_name="rf_classification_report.csv",
            mime="text/csv",
        )
    else:
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

# 여러 파일 업로드 (Multiple file uploader)
uploaded_files = st.file_uploader(texts["upload"], type=["wav"], accept_multiple_files=True)

if uploaded_files:
    filenames = [file.name for file in uploaded_files]
    selected_file = st.selectbox(texts["select_file"], filenames)

    # 선택한 파일 객체 찾기 (Find selected file object)
    file_obj = next(file for file in uploaded_files if file.name == selected_file)

    try:
        # 오디오 재생 (Play audio)
        audio_bytes = file_obj.read()
        st.audio(audio_bytes, format='audio/wav')
        file_obj.seek(0)  # 파일 포인터 초기화 (Reset file pointer)

        # MFCC 특징 추출 (Extract MFCC features)
        y, sr = librosa.load(file_obj)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=29) #n_mfcc= 13 → 29로 변경
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features = np.concatenate((mfcc_mean, mfcc_std)).reshape(1, -1)

        # 스케일링 (Scaling features)
        features_scaled = scaler.transform(features)

        # 예측 수행 (Make prediction)
        prediction_encoded = model.predict(features_scaled)
        prediction = label_encoder.inverse_transform(prediction_encoded)
        st.success(f"🎶 **{texts['predicted_genre']}:** `{prediction[0].capitalize()}`")

        # 예측 확률 보기 (가능할 경우) (Show prediction probabilities if available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            classes_encoded = model.classes_
            classes = label_encoder.inverse_transform(classes_encoded)
            proba_dict = dict(zip(classes, proba))
            st.markdown("### 🔍 Prediction Probabilities")
            st.bar_chart(proba_dict)

        # 정확도 요약 (Accuracy summary)
        with st.expander(texts["accuracy_summary"]):
            st.markdown(f"""
            - **{texts['accuracy_rf']}:** ~64%  
            - **{texts['accuracy_svm']}:** ~61%  
            - {texts['best_genres']}: 🎼 `Classical`, 🤘 `Metal`, 🎷 `Jazz`
            """)

        # MFCC 히트맵 시각화 (Visualize MFCC heatmap)
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

# --- 실시간 마이크 녹음 기능 (Real-Time Mic Recording) ---

st.markdown("## 🎤 Real-Time Mic Recording")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        super().__init__()
        self.recorded_frames = []  # 녹음된 프레임 저장 (Store recorded frames)

    def recv(self, frame):
        self.recorded_frames.append(frame.to_ndarray())  # 프레임 받아서 저장 (Append received frame)
        return frame  # 프레임 그대로 반환 (Return frame as is)

    def save_recording(self, filename="recorded_audio.wav", samplerate=16000):
        audio = np.concatenate(self.recorded_frames, axis=0)  # 녹음된 프레임 합치기 (Concatenate frames)
        sf.write(filename, audio, samplerate)  # wav 파일로 저장 (Save as wav file)
        return filename

    def get_audio_data(self):
        return np.concatenate(self.recorded_frames, axis=0)  # 녹음된 오디오 numpy 배열 반환 (Return recorded audio as numpy array)

ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,  # 오디오 프로세서 지정 (Assign audio processor)
)

if ctx:
    if ctx.state.playing:
        st.info("🎙 Recording... Click STOP when done.")  # 녹음 중 안내 (Recording info)
    elif not ctx.state.playing and hasattr(ctx, "processor") and ctx.processor:
        try:
            st.success("Recording complete! Analyzing...")  # 녹음 완료 안내 (Recording complete info)
            
            # 녹음된 오디오 numpy 배열 (Recorded audio numpy array)
            audio_np = np.concatenate(ctx.processor.recorded_frames, axis=0)
            
            # WebRTC 기본 샘플레이트 (일반적으로 48000) (WebRTC default sample rate)
            samplerate = 48000
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(tmpfile.name, audio_np, samplerate)  # 임시 wav 파일 저장 (Save temp wav file)
                
                # librosa로 로드 및 MFCC 추출 (Load audio and extract MFCC with librosa)
                y, sr = librosa.load(tmpfile.name)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=29) # n_mfcc = 13 → 29로 변경
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                features = np.concatenate((mfcc_mean, mfcc_std)).reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                # 모델 예측 (Model prediction)
                prediction_encoded = model.predict(features_scaled)
                prediction = label_encoder.inverse_transform(prediction_encoded)[0]
                st.success(f"🎶 Predicted Genre (Mic): `{prediction.capitalize()}`")
                
                # 예측 확률 시각화 (Show prediction probabilities)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features_scaled)[0]
                    classes = label_encoder.inverse_transform(model.classes_)
                    st.bar_chart(dict(zip(classes, proba)))
                
                # MFCC 히트맵 표시 (Show MFCC heatmap)
                if st.checkbox("Show MFCC Heatmap (Mic Input)"):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
                    ax.set_title("MFCC Features (Mic Input)")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("MFCC Coefficients")
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during microphone processing: {e}")
    else:
        # 녹음 시작 안내 문구 출력 (Show mic recording start info)
        st.info(texts["mic_start_info"])
else:
    # ctx 객체가 없을 때도 안내 메시지 출력 (Show mic start info if ctx is None)
    st.info(texts["mic_start_info"])
    
