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
from io import BytesIO
import tensorflow as tf

# --- 다국어 딕셔너리 (Languages dictionary) ---
lang_dict = {
    "ko": {
        "language_name": "Korean (한국어)",
        "title": "🎵 음악 장르 분류기 (CNN 지원 포함)",
        "upload": ".wav 파일을 업로드하세요",
        "select_model": "모델 선택",
        "test_file": "🎧 테스트 파일:",
        "download": "⬇️ 테스트용 .wav 파일 다운로드",
        "prediction": "🎶 예측된 장르:",
        "prob": "### 🔍 예측 확률",
        "mfcc": "MFCC 히트맵 보기",
        "mel": "Mel 스펙트로그램 보기",
        "mfcc_title": "MFCC 특징",
        "mel_title": "Mel 스펙트로그램",
        "error": "❌ 예측 중 오류 발생",
        "upload_prompt": ".wav 파일을 업로드하여 시작하세요.",
        "no_file": "테스트용 .wav 파일이 없습니다.",
        "change_test": "🔄 테스트 파일 변경"
    },
    "en": {
        "language_name": "English (영어)",
        "title": "🎵 Music Genre Classifier (with CNN support)",
        "upload": "Upload a .wav file",
        "select_model": "Select a model",
        "test_file": "🎧 Test file:",
        "download": "⬇️ Download test .wav file",
        "prediction": "🎶 Predicted Genre:",
        "prob": "### 🔍 Prediction Probabilities",
        "mfcc": "Show MFCC Heatmap",
        "mel": "Show Mel Spectrogram",
        "mfcc_title": "MFCC Features",
        "mel_title": "Mel Spectrogram",
        "error": "❌ Error during prediction",
        "upload_prompt": "Please upload a .wav file to get started.",
        "no_file": "No test .wav file found.",
        "change_test": "🔄 Change test file"
    },
    "de": {
        "language_name": "Deutsch (German-독일어)",
        "title": "🎵 Musikgenre-Klassifizierer (mit CNN-Unterstützung)",
        "upload": "Lade eine .wav-Datei hoch",
        "select_model": "Modell auswählen",
        "test_file": "🎧 Testdatei:",
        "download": "⬇️ Test-.wav-Datei herunterladen",
        "prediction": "🎶 Vorhergesagtes Genre:",
        "prob": "### 🔍 Vorhersagewahrscheinlichkeiten",
        "mfcc": "MFCC-Heatmap anzeigen",
        "mel": "Mel-Spektrogramm anzeigen",
        "mfcc_title": "MFCC-Merkmale",
        "mel_title": "Mel-Spektrogramm",
        "error": "❌ Fehler bei der Vorhersage",
        "upload_prompt": "Bitte lade eine .wav-Datei hoch, um zu starten.",
        "no_file": "Keine Test-.wav-Datei gefunden.",
        "change_test": "🔄 Testdatei wechseln"
    },
    "pl": {
        "language_name": "Polski (Polish-폴란드어)",
        "title": "🎵 Klasyfikator gatunków muzycznych (z obsługą CNN)",
        "upload": "Prześlij plik .wav",
        "select_model": "Wybierz model",
        "test_file": "🎧 Plik testowy:",
        "download": "⬇️ Pobierz plik .wav do testów",
        "prediction": "🎶 Przewidywany gatunek:",
        "prob": "### 🔍 Prawdopodobieństwa przewidywania",
        "mfcc": "Pokaż mapę cieplną MFCC",
        "mel": "Pokaż spektrogram Mel",
        "mfcc_title": "Cechy MFCC",
        "mel_title": "Spektrogram Mel",
        "error": "❌ Błąd podczas przewidywania",
        "upload_prompt": "Prześlij plik .wav, aby rozpocząć.",
        "no_file": "Nie znaleziono pliku .wav do testów.",
        "change_test": "🔄 Zmień plik testowy"
    },
    "hi": {
        "language_name": "हिन्दी (Hindi-인도-힌디어)",        
        "title": "🎵 म्यूजिक शैली वर्गीकरण (CNN समर्थन सहित)",
        "upload": ".wav फ़ाइल अपलोड करें",
        "select_model": "मॉडल चुनें",
        "test_file": "🎧 परीक्षण फ़ाइल:",
        "download": "⬇️ परीक्षण .wav फ़ाइल डाउनलोड करें",
        "prediction": "🎶 अनुमानित शैली:",
        "prob": "### 🔍 भविष्यवाणी की संभावनाएँ",
        "mfcc": "MFCC हीटमैप दिखाएँ",
        "mel": "Mel स्पेक्ट्रोग्राम दिखाएँ",
        "mfcc_title": "MFCC विशेषताएँ",
        "mel_title": "Mel स्पेक्ट्रोग्राम",
        "error": "❌ पूर्वानुमान के दौरान त्रुटि",
        "upload_prompt": "शुरू करने के लिए कृपया .wav फ़ाइल अपलोड करें।",
        "no_file": "कोई परीक्षण .wav फ़ाइल नहीं मिली।",
        "change_test": "🔄 परीक्षण फ़ाइल बदलें"
    },
    "ta": {
        "language_name": "தமிழ் (Tamil-인도-타말어)",
        "title": "🎵 இசை வகை வகைப்படுத்தி (CNN ஆதரவுடன்)",
        "upload": ".wav கோப்பை பதிவேற்று",
        "select_model": "மாதிரியை தேர்ந்தெடு",
        "test_file": "🎧 சோதனை கோப்பு:",
        "download": "⬇️ சோதனை .wav கோப்பை பதிவிறக்கு",
        "prediction": "🎶 கணிக்கப்பட்ட இசை வகை:",
        "prob": "### 🔍 கணிப்பு சாத்தியக்கூறுகள்",
        "mfcc": "MFCC வெப்பப்படத்தை காட்டு",
        "mel": "Mel ஸ்பெக்ட்ரோகிராமை காட்டு",
        "mfcc_title": "MFCC அம்சங்கள்",
        "mel_title": "Mel ஸ்பெக்ட்ரோகம்",
        "error": "❌ கணிப்பில் பிழை ஏற்பட்டது",
        "upload_prompt": "தொடங்க .wav கோப்பை பதிவேற்று.",
        "no_file": ".wav சோதனை கோப்பு இல்லை.",
        "change_test": "🔄 சோதனை கோப்பை மாற்று"
    },
    "zh": {
        "language_name": "中文 (China-중국어)",
        "title": "🎵 音乐流派分类器（支持CNN）",
        "upload": "上传 .wav 文件",
        "select_model": "选择模型",
        "test_file": "🎧 测试文件:",
        "download": "⬇️ 下载测试 .wav 文件",
        "prediction": "🎶 预测的流派:",
        "prob": "### 🔍 预测概率",
        "mfcc": "显示 MFCC 热图",
        "mel": "显示 Mel 频谱图",
        "mfcc_title": "MFCC 特征",
        "mel_title": "Mel 频谱图",
        "error": "❌ 预测时发生错误",
        "upload_prompt": "请上传 .wav 文件以开始。",
        "no_file": "未找到测试 .wav 文件。",
        "change_test": "🔄 更换测试文件"
    },
    "yue": {
        "language_name": "粵語 (Cantonese-홍콩어)",
        "title": "🎵 音樂類型分類器（支援CNN）",
        "upload": "上傳 .wav 檔案",
        "select_model": "選擇模型",
        "test_file": "🎧 測試檔案：",
        "download": "⬇️ 下載測試 .wav 檔案",
        "prediction": "🎶 預測的類型：",
        "prob": "### 🔍 預測機率",
        "mfcc": "顯示 MFCC 熱圖",
        "mel": "顯示 Mel 頻譜圖",
        "mfcc_title": "MFCC 特徵",
        "mel_title": "Mel 頻譜圖",
        "error": "❌ 預測時發生錯誤",
        "upload_prompt": "請上傳 .wav 檔案以開始。",
        "no_file": "未找到測試用的 .wav 檔案。",
        "change_test": "🔄 更換測試檔案"
    },
    "ja": {
        "language_name": "日本語 (Japanese-일본어)",
        "title": "🎵 音楽ジャンル分類器（CNN対応）",
        "upload": ".wavファイルをアップロード",
        "select_model": "モデルを選択",
        "test_file": "🎧 テストファイル：",
        "download": "⬇️ テスト用 .wav ファイルをダウンロード",
        "prediction": "🎶 予測されたジャンル：",
        "prob": "### 🔍 予測確率",
        "mfcc": "MFCC ヒートマップを表示",
        "mel": "Mel スペクトログラムを表示",
        "mfcc_title": "MFCC 特徴",
        "mel_title": "Mel スペクトログラム",
        "error": "❌ 予測中にエラーが発生しました",
        "upload_prompt": ".wav ファイルをアップロードして開始してください。",
        "no_file": "テスト用の .wav ファイルが見つかりません。",
        "change_test": "🔄 テストファイルを変更"
    },
    "fr": {
        "language_name": "Français (Franch-프랑스어)",
        "title": "🎵 Classificateur de genre musical (avec support CNN)",
        "upload": "Téléversez un fichier .wav",
        "select_model": "Choisissez un modèle",
        "test_file": "🎧 Fichier de test :",
        "download": "⬇️ Télécharger le fichier .wav de test",
        "prediction": "🎶 Genre prédit :",
        "prob": "### 🔍 Probabilités de prédiction",
        "mfcc": "Afficher la carte thermique MFCC",
        "mel": "Afficher le spectrogramme Mel",
        "mfcc_title": "Caractéristiques MFCC",
        "mel_title": "Spectrogramme Mel",
        "error": "❌ Erreur lors de la prédiction",
        "upload_prompt": "Veuillez téléverser un fichier .wav pour commencer.",
        "no_file": "Aucun fichier .wav de test trouvé.",
        "change_test": "🔄 Changer de fichier de test"
    },
    "it": {
        "language_name": "Italiano (Italian-이탈리아어)",
        "title": "🎵 Classificatore di generi musicali (con supporto CNN)",
        "upload": "Carica un file .wav",
        "select_model": "Seleziona un modello",
        "test_file": "🎧 File di test:",
        "download": "⬇️ Scarica file .wav di test",
        "prediction": "🎶 Genere previsto:",
        "prob": "### 🔍 Probabilità di previsione",
        "mfcc": "Mostra la mappa di calore MFCC",
        "mel": "Mostra lo spettrogramma Mel",
        "mfcc_title": "Caratteristiche MFCC",
        "mel_title": "Spettrogramma Mel",
        "error": "❌ Errore durante la previsione",
        "upload_prompt": "Carica un file .wav per iniziare.",
        "no_file": "File .wav di test non trovato.",
        "change_test": "🔄 Cambia file di test"
    },
    "ru": {
        "language_name": "Русский (Russian-러시아어)",
        "title": "🎵 Классификатор музыкальных жанров (с поддержкой CNN)",
        "upload": "Загрузите .wav файл",
        "select_model": "Выберите модель",
        "test_file": "🎧 Тестовый файл:",
        "download": "⬇️ Скачать тестовый .wav файл",
        "prediction": "🎶 Предсказанный жанр:",
        "prob": "### 🔍 Вероятности предсказания",
        "mfcc": "Показать тепловую карту MFCC",
        "mel": "Показать спектрограмму Mel",
        "mfcc_title": "Признаки MFCC",
        "mel_title": "Спектрограмма Mel",
        "error": "❌ Ошибка при предсказании",
        "upload_prompt": "Пожалуйста, загрузите .wav файл для начала.",
        "no_file": "Тестовый .wav файл не найден.",
        "change_test": "🔄 Сменить тестовый файл"
    },
    "es": {
        "language_name": "Español (Spanish-스페인어)",
        "title": "🎵 Clasificador de géneros musicales (con soporte CNN)",
        "upload": "Sube un archivo .wav",
        "select_model": "Selecciona un modelo",
        "test_file": "🎧 Archivo de prueba:",
        "download": "⬇️ Descargar archivo .wav de prueba",
        "prediction": "🎶 Género predicho:",
        "prob": "### 🔍 Probabilidades de predicción",
        "mfcc": "Mostrar mapa de calor MFCC",
        "mel": "Mostrar espectrograma Mel",
        "mfcc_title": "Características MFCC",
        "mel_title": "Espectrograma Mel",
        "error": "❌ Error durante la predicción",
        "upload_prompt": "Por favor, sube un archivo .wav para empezar.",
        "no_file": "No se encontró archivo .wav de prueba.",
        "change_test": "🔄 Cambiar archivo de prueba"
    },
    "ar": {
        "language_name": "العربية (Arabic-아랍어)",
        "title": "🎵 مصنف نوع الموسيقى (بدعم من CNN)",
        "upload": "قم بتحميل ملف .wav",
        "select_model": "اختر نموذجًا",
        "test_file": "🎧 ملف الاختبار:",
        "download": "⬇️ تنزيل ملف .wav للاختبار",
        "prediction": "🎶 النوع المتوقع:",
        "prob": "### 🔍 احتمالات التنبؤ",
        "mfcc": "عرض خريطة الحرارة MFCC",
        "mel": "عرض طيف Mel",
        "mfcc_title": "ميزات MFCC",
        "mel_title": "طيف Mel",
        "error": "❌ خطأ أثناء التنبؤ",
        "upload_prompt": "يرجى تحميل ملف .wav للبدء.",
        "no_file": "لم يتم العثور على ملف .wav للاختبار.",
        "change_test": "🔄 تغيير ملف الاختبار"
    },
    "pt": {
        "language_name": "Português (Portuguese-포르투갈어)",
        "title": "🎵 Classificador de Gêneros Musicais (com suporte CNN)",
        "upload": "Envie um arquivo .wav",
        "select_model": "Escolha um modelo",
        "test_file": "🎧 Arquivo de teste:",
        "download": "⬇️ Baixar arquivo .wav de teste",
        "prediction": "🎶 Gênero previsto:",
        "prob": "### 🔍 Probabilidades de previsão",
        "mfcc": "Mostrar mapa de calor MFCC",
        "mel": "Mostrar espectrograma Mel",
        "mfcc_title": "Características MFCC",
        "mel_title": "Espectrograma Mel",
        "error": "❌ Erro durante a previsão",
        "upload_prompt": "Por favor, envie um arquivo .wav para começar.",
        "no_file": "Arquivo .wav de teste não encontrado.",
        "change_test": "🔄 Alterar arquivo de teste"
    },
    "vi": {
        "language_name": "Tiếng Việt (Vietnamese-베트남어)",
        "title": "🎵 Bộ phân loại thể loại nhạc (hỗ trợ CNN)",
        "upload": "Tải lên tệp .wav",
        "select_model": "Chọn mô hình",
        "test_file": "🎧 Tệp kiểm tra:",
        "download": "⬇️ Tải xuống tệp .wav kiểm tra",
        "prediction": "🎶 Thể loại dự đoán:",
        "prob": "### 🔍 Xác suất dự đoán",
        "mfcc": "Hiển thị bản đồ nhiệt MFCC",
        "mel": "Hiển thị phổ Mel",
        "mfcc_title": "Đặc trưng MFCC",
        "mel_title": "Phổ Mel",
        "error": "❌ Lỗi trong quá trình dự đoán",
        "upload_prompt": "Vui lòng tải lên tệp .wav để bắt đầu.",
        "no_file": "Không tìm thấy tệp .wav kiểm tra.",
        "change_test": "🔄 Đổi tệp kiểm tra khác"
    },
    "tr": {
        "language_name": "Türkçe (Turkish-튀르키예어)",
        "title": "🎵 Müzik Türü Sınıflandırıcı (CNN desteği ile)",
        "upload": ".wav dosyası yükleyin",
        "select_model": "Bir model seçin",
        "test_file": "🎧 Test dosyası:",
        "download": "⬇️ Test .wav dosyasını indir",
        "prediction": "🎶 Tahmin Edilen Tür:",
        "prob": "### 🔍 Tahmin Olasılıkları",
        "mfcc": "MFCC Isı Haritasını Göster",
        "mel": "Mel Spektrogramını Göster",
        "mfcc_title": "MFCC Özellikleri",
        "mel_title": "Mel Spektrogramı",
        "error": "❌ Tahmin sırasında hata oluştu",
        "upload_prompt": "Başlamak için lütfen bir .wav dosyası yükleyin.",
        "no_file": "Test .wav dosyası bulunamadı.",
        "change_test": "🔄 Test dosyasını değiştir"
    },
}

# --- 장르 레이블 (CNN용)
genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']

# --- 상태 초기화 ---
if "refresh_sample" not in st.session_state:
    st.session_state.refresh_sample = False

# --- 언어 선택 ---
selected_lang = st.sidebar.selectbox("Language / 언어", options=list(lang_dict.keys()), format_func=lambda x: lang_dict[x]["language_name"])
texts = lang_dict[selected_lang]

# --- 무작위 wav 파일 선택 ---
def pick_random_wav_file(base_dir="/content/gtzan_data/Data/genres_original"):
    genres = [g for g in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, g))]
    random_genre = random.choice(genres)
    genre_path = os.path.join(base_dir, random_genre)
    wav_files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
    if not wav_files:
        return None, None
    random_file = random.choice(wav_files)
    return os.path.join(genre_path, random_file), f"{random_genre} - {random_file}"

# --- 오디오 다운로드 링크 생성 ---
def get_audio_download_link(file_path, label):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:audio/wav;base64,{b64}" download="test_sample.wav">{label}</a>'

# --- CNN 모델 로드 ---
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model(MODEL_FILES["CNN"], compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 기존 모델 로드 ---
def load_model_files(model_name: str):
    model_path = os.path.join(BASE_PATH, MODEL_FILES[model_name])
    scaler_path = os.path.join(BASE_PATH, SCALER_FILE)
    label_enc_path = os.path.join(BASE_PATH, LABEL_ENCODER_FILE)
    report_path = os.path.join(BASE_PATH, REPORT_FILES[model_name])

    for path in [model_path, scaler_path, label_enc_path, report_path]:
        if not os.path.isfile(path):
            st.error(f"Required file not found: {path}")
            st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_enc_path)
    report_df = pd.read_csv(report_path, index_col=0)
    with open(report_path, "rb") as f:
        report_data = f.read()

    return model, scaler, label_encoder, report_df, report_data, report_path

# --- 특징 추출 함수들 ---
def extract_features(audio_bytes, n_mfcc):
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

# --- 클래스 라벨 정렬 확인 ---
def check_class_alignment(model, label_encoder):
    try:
        return label_encoder.inverse_transform(model.classes_)
    except Exception:
        return label_encoder.classes_

# --- 상수 설정 ---
BASE_PATH = ""
N_MFCC = 13
SAMPLE_AUDIO_FILE = "sample.wav"
MODEL_FILES = {"Random Forest": "rf_model.pkl", "SVM": "svm_model.pkl", "CNN": "cnn_genre_model.keras"}
REPORT_FILES = {"Random Forest": "rf_classification_report.csv", "SVM": "svm_classification_report.csv"}
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# --- Streamlit UI 시작 ---
st.set_page_config(page_title="Music Genre Classifier", layout="centered")
st.title(texts["title"])

model_option = st.selectbox(texts["select_model"], list(MODEL_FILES.keys()))

if st.sidebar.button(texts["change_test"]):
    st.session_state.refresh_sample = not st.session_state.refresh_sample

sample_path, sample_name = pick_random_wav_file()
if sample_path:
    st.sidebar.markdown(f"{texts['test_file']} `{sample_name}`")
    st.sidebar.markdown(get_audio_download_link(sample_path, texts['download']), unsafe_allow_html=True)
else:
    st.sidebar.warning(texts["no_file"])

if model_option == "CNN":
    model = load_cnn_model()
else:
    model, scaler, label_encoder, _, _, _ = load_model_files(model_option)
    model_classes = check_class_alignment(model, label_encoder)

uploaded_file = st.file_uploader(texts["upload"], type=["wav"])

if uploaded_file:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

    if model_option == "CNN":
        features, mel = extract_mel_spectrogram(audio_bytes)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_label = genre_labels[predicted_index]

        st.success(f"{texts['prediction']} `{predicted_label.capitalize()}`")
        st.markdown(texts["prob"])
        st.bar_chart(dict(zip(genre_labels, prediction[0])))

        if st.checkbox(texts["mel"]):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mel, cmap="YlGnBu", ax=ax)
            ax.set(title=texts["mel_title"], xlabel="Time", ylabel="Mel Bands")
            st.pyplot(fig)
            plt.close(fig)

    else:
        try:
            features, mfcc = extract_features(audio_bytes, N_MFCC)
            features_scaled = scaler.transform(features)
            prediction_encoded = model.predict(features_scaled)
            prediction = label_encoder.inverse_transform(prediction_encoded)

            st.success(f"{texts['prediction']} `{prediction[0].capitalize()}`")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_scaled)[0]
                st.markdown(texts["prob"])
                st.bar_chart(dict(zip(model_classes, proba)))

            if st.checkbox(texts["mfcc"]):
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
                ax.set(title=texts["mfcc_title"], xlabel="Time", ylabel="MFCC Coefficients")
                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.error(texts["error"])
            st.exception(e)
else:
    st.info(texts["upload_prompt"])
