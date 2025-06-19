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

# --- 다국어 딕셔너리 (Languages dictionary) ---
lang_dict = {
     "ko": {
        "language_name": "Korean (한국어)",
        "title": "음악 장르 분류기",
        "upload": ".wav 파일을 업로드하세요",
        "select_model": "모델 선택",
        "download_rf": "⬇️ 랜덤 포레스트 분류 리포트 다운로드",
        "download_svm": "⬇️ SVM 분류 리포트 다운로드",
        "show_heatmap_mic": "MFCC 히트맵 보기 (마이크 입력)",
        "mfcc_heatmap_title_mic": "MFCC 특징 (마이크 입력)",
        "predicted_genre": "예측된 장르",
        "show_heatmap": "MFCC 히트맵 보기",
        "accuracy_summary": "모델 정확도 요약",
        "accuracy_rf": "랜덤 포레스트 정확도",
        "accuracy_svm": "SVM 정확도",
        "best_genres": "가장 높은 성능을 보이는 장르",
        "about_app": "앱 정보",
        "model_performance": "모델 성능 지표",
        "select_file": "분류할 파일 선택",
        "choose_language": "언어 선택 / Choose Language",
        "start_info": "하나 이상의 .wav 파일을 업로드 해주세요.",
        "mic_start_info": "녹음을 시작하려면 위 버튼을 클릭하세요.",
        "model_desc_rf": "랜덤 포레스트: 여러 판단 기준을 모아 최종 결정을 내리는 방법",
        "model_desc_svm": "SVM: 데이터 경계선을 찾아 구분하는 방법"
    },
    "en": {
        "language_name": "English (영어)",
        "title": "Music Genre Classifier",
        "upload": "Upload one or more .wav files",
        "select_model": "Choose a model",
        "download_rf": "⬇️ Download Random Forest Classification Report",
        "download_svm": "⬇️ Download SVM Classification Report",
        "show_heatmap_mic": "Show MFCC Heatmap (Mic Input)",
        "mfcc_heatmap_title_mic": "MFCC Features (Mic Input)",
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
        "mic_start_info": "Click the button above to start recording.",
        "model_desc_rf": "Random Forest: A method that makes the final decision by combining many simple decisions",
        "model_desc_svm": "SVM: A method that finds the boundary line to separate different groups of data"
    },
    "de": {
        "language_name": "Deutsch (German-독일어)",
        "title": "Musikgenre-Klassifikator",
        "upload": "Laden Sie eine oder mehrere .wav-Dateien hoch",
        "select_model": "Wählen Sie ein Modell",
        "download_rf": "⬇️ Random Forest Klassifikationsbericht herunterladen",
        "download_svm": "⬇️ SVM Klassifikationsbericht herunterladen",
        "show_heatmap_mic": "MFCC Heatmap anzeigen (Mikrofoneingabe)",
        "mfcc_heatmap_title_mic": "MFCC Merkmale (Mikrofoneingabe)",
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
        "mic_start_info": "Klicken Sie oben auf die Schaltfläche, um die Aufnahme zu starten.",
        "model_desc_rf": "Random Forest: Eine Methode, die eine endgültige Entscheidung trifft, indem sie viele einfache Entscheidungen kombiniert",
        "model_desc_svm": "SVM: Eine Methode, die die Grenze findet, um verschiedene Datenmengen zu trennen"
    },
    "pl": {
        "language_name": "Polski (Polish-폴란드어)",
        "title": "Klasyfikator gatunków muzycznych",
        "upload": "Prześlij jeden lub więcej plików .wav",
        "select_model": "Wybierz model",
        "download_rf": "⬇️ Pobierz raport klasyfikacji Random Forest",
        "download_svm": "⬇️ Pobierz raport klasyfikacji SVM",
        "show_heatmap_mic": "Pokaż mapę ciepła MFCC (wejście z mikrofonu)",
        "mfcc_heatmap_title_mic": "Cechy MFCC (wejście z mikrofonu)",
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
        "mic_start_info": "Kliknij przycisk powyżej, aby rozpocząć nagrywanie.",
        "model_desc_rf": "Random Forest: Metoda podejmująca ostateczną decyzję poprzez połączenie wielu prostych decyzji",
        "model_desc_svm": "SVM: Metoda znajdująca linię graniczną rozdzielającą różne grupy danych"
    },
    "hi": {
        "language_name": "हिन्दी (Hindi-인도-힌두어)",        
        "title": "संगीत शैली वर्गीकर्ता",
        "upload": ".wav फ़ाइल अपलोड करें",
        "select_model": "मॉडल चुनें",
        "download_rf": "⬇️ रैंडम फॉरेस्ट वर्गीकरण रिपोर्ट डाउनलोड करें",
        "download_svm": "⬇️ एसवीएम वर्गीकरण रिपोर्ट डाउनलोड करें",
        "show_heatmap_mic": "MFCC हीटमैप दिखाएँ (माइक इनपुट)",
        "mfcc_heatmap_title_mic": "MFCC फीचर्स (माइक इनपुट)",
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
        "mic_start_info": "रिकॉर्डिंग शुरू करने के लिए ऊपर दिए गए बटन पर क्लिक करें।",
        "model_desc_rf": "Random Forest: कई सरल निर्णयों को मिलाकर अंतिम निर्णय लेने की विधि",
        "model_desc_svm": "SVM: डेटा समूहों को अलग करने वाली सीमा रेखा खोजने की विधि"
    },
    "ta": {
        "language_name": "தமிழ் (Tamil-인도-타말어)",
        "title": "பாடல் வகை வகைப்பான்",
        "upload": ".wav கோப்புகளை பதிவேற்றவும்",
        "select_model": "மாதிரியைத் தேர்ந்தெடுக்கவும்",
        "download_rf": "⬇️ ரேண்டம் ஃபாரெஸ்ட் வகைப்பாட்டு அறிக்கை பதிவிறக்கு",
        "download_svm": "⬇️ எஸ்விஎம் வகைப்பாட்டு அறிக்கை பதிவிறக்கு",
        "show_heatmap_mic": "MFCC ஹீட்மாப் காண்க (மைக்ரோபோன் உள்ளீடு)",
        "mfcc_heatmap_title_mic": "MFCC அம்சங்கள் (மைக்ரோபோன் உள்ளீடு)",
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
        "mic_start_info": "பதிவு செய்ய ஆரம்பிக்க மேலுள்ள பொத்தானை அழுத்தவும்.",
        "model_desc_rf": "Random Forest: பல எளிய முடிவுகளை இணைத்து இறுதி முடிவை எடுக்கும் முறை",
        "model_desc_svm": "SVM: தரவு குழுக்களை பிரிக்க எல்லை வரியை கண்டுபிடிக்கும் முறை"
    },
    "zh": {
        "language_name": "中文 (China-중국어)",        
        "title": "音乐类别分类器",
        "upload": "上传一个或多个.wav文件",
        "select_model": "选择模型",
        "download_rf": "⬇️ 下载随机森林分类报告",
        "download_svm": "⬇️ 下载SVM分类报告",
        "show_heatmap_mic": "显示MFCC热图（麦克风输入）",
        "mfcc_heatmap_title_mic": "MFCC 特征（麦克风输入）",
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
        "mic_start_info": "点击上方按钮开始录音。",
        "model_desc_rf": "随机森林：通过结合多个简单决策来做出最终决定的方法",
        "model_desc_svm": "支持向量机：寻找分割不同数据组的边界线的方法"
    },
    "hk": {
        "language_name": "繁體中文-香港粵語 (Hong Kong Cantonese-홍콩어)",
        "title": "音樂類型分類器",
        "upload": "上載一個或多個.wav檔案",
        "select_model": "選擇模型",
        "download_rf": "⬇️ 下載隨機森林分類報告",
        "download_svm": "⬇️ 下載SVM分類報告",
        "show_heatmap_mic": "顯示MFCC熱圖（麥克風輸入）",
        "mfcc_heatmap_title_mic": "MFCC 特徵（麥克風輸入）",
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
        "mic_start_info": "點擊上方按鈕開始錄音。",
        "model_desc_rf": "隨機森林：透過結合多個簡單決策來作出最終決定的方法",
        "model_desc_svm": "支持向量機：尋找分隔不同數據組的邊界線的方法"
    },
    "ja": {
        "language_name": "日本語 (Japanese-일본어)",
        "title": "音楽ジャンル分類器",
        "upload": "1つ以上の.wavファイルをアップロードしてください",
        "select_model": "モデルを選択",
        "download_rf": "⬇️ ランダムフォレスト分類レポートをダウンロード",
        "download_svm": "⬇️ SVM分類レポートをダウンロード",
        "show_heatmap_mic": "MFCCヒートマップを表示（マイク入力）",
        "mfcc_heatmap_title_mic": "MFCC 特徴（マイク入力）",
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
        "mic_start_info": "録音を開始するには上のボタンをクリックしてください。",
        "model_desc_rf": "ランダムフォレスト：多数の単純な判断を組み合わせて最終決定を行う方法",
        "model_desc_svm": "SVM：異なるデータ群を分ける境界線を見つける方法"
    },
    "fr": {
        "language_name": "Français (Franch-프랑스어)",
        "title": "Classificateur de genre musical",
        "upload": "Téléchargez un ou plusieurs fichiers .wav",
        "select_model": "Choisir un modèle",
        "download_rf": "⬇️ Télécharger le rapport de classification Random Forest",
        "download_svm": "⬇️ Télécharger le rapport de classification SVM",
        "show_heatmap_mic": "Afficher la carte thermique MFCC (entrée micro)",
        "mfcc_heatmap_title_mic": "Caractéristiques MFCC (entrée micro)",
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
        "mic_start_info": "Cliquez sur le bouton ci-dessus pour commencer l'enregistrement.",
        "model_desc_rf": "Forêt Aléatoire : Une méthode qui prend la décision finale en combinant de nombreuses décisions simples",
        "model_desc_svm": "SVM : Une méthode qui trouve la ligne de séparation pour distinguer différents groupes de données"
    },
    "it": {
        "language_name": "Italiano (Italian-이태리어)",
        "title": "Classificatore di genere musicale",
        "upload": "Carica uno o più file .wav",
        "select_model": "Scegli un modello",
        "download_rf": "⬇️ Scarica il rapporto di classificazione Random Forest",
        "download_svm": "⬇️ Scarica il rapporto di classificazione SVM",
        "show_heatmap_mic": "Mostra la mappa di calore MFCC (input microfono)",
        "mfcc_heatmap_title_mic": "Caratteristiche MFCC (input microfono)",
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
        "mic_start_info": "Fai clic sul pulsante sopra per iniziare la registrazione.",
        "model_desc_rf": "Random Forest: Un metodo che prende la decisione finale combinando molte decisioni semplici",
        "model_desc_svm": "SVM: Un metodo che trova la linea di confine per separare diversi gruppi di dati"
    },
    "ru": {
        "language_name": "Русский (Russian-러시아어)",
        "title": "Классификатор музыкальных жанров",
        "upload": "Загрузите один или несколько файлов .wav",
        "select_model": "Выберите модель",
        "download_rf": "⬇️ Скачать отчет классификации Random Forest",
        "download_svm": "⬇️ Скачать отчет классификации SVM",
        "show_heatmap_mic": "Показать тепловую карту MFCC (вход с микрофона)",
        "mfcc_heatmap_title_mic": "Признаки MFCC (вход с микрофона)",
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
        "mic_start_info": "Нажмите кнопку выше, чтобы начать запись.",
        "model_desc_rf": "Случайный лес: метод, который принимает окончательное решение, объединяя множество простых решений",
        "model_desc_svm": "SVM: метод, который находит границу для разделения различных групп данных"
    },
    "es": {
        "language_name": "Español (Spanish-에스파냐어)",
        "title": "Clasificador de Géneros Musicales",
        "upload": "Sube uno o más archivos .wav",
        "select_model": "Elige un modelo",
        "download_rf": "⬇️ Descargar informe de clasificación Random Forest",
        "download_svm": "⬇️ Descargar informe de clasificación SVM",
        "show_heatmap_mic": "Mostrar mapa de calor MFCC (entrada de micrófono)",
        "mfcc_heatmap_title_mic": "Características MFCC (entrada de micrófono)",
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
        "mic_start_info": "Haga clic en el botón de arriba para comenzar la grabación.",
        "model_desc_rf": "Bosque Aleatorio: Un método que toma la decisión final combinando muchas decisiones simples",
        "model_desc_svm": "SVM: Un método que encuentra la línea límite para separar diferentes grupos de datos"
    },
    "ar": {
        "language_name": "العربية (Arabic-아랍어)",
        "title": "مصنف نوع الموسيقى",
        "upload": "قم بتحميل ملف أو أكثر بصيغة .wav",
        "select_model": "اختر نموذجًا",
        "download_rf": "⬇️ تحميل تقرير تصنيف Random Forest",
        "download_svm": "⬇️ تحميل تقرير تصنيف SVM",
        "show_heatmap_mic": "عرض خريطة الحرارة MFCC (إدخال الميكروفون)",
        "mfcc_heatmap_title_mic": "ميزات MFCC (إدخال الميكروفون)",
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
        "mic_start_info": "انقر فوق الزر أعلاه لبدء التسجيل.",
        "model_desc_rf": "الغابة العشوائية: طريقة تتخذ القرار النهائي عن طريق دمج العديد من القرارات البسيطة",
        "model_desc_svm": "SVM: طريقة تجد خط الحدود لفصل مجموعات البيانات المختلفة"
    },
    "pt": {
        "language_name": "Português (Portuguese-포르투갈어)",
        "title": "Classificador de Gêneros Musicais",
        "upload": "Faça upload de um ou mais arquivos .wav",
        "select_model": "Escolha um modelo",
        "download_rf": "⬇️ Baixar relatório de classificação Random Forest",
        "download_svm": "⬇️ Baixar relatório de classificação SVM",
        "show_heatmap_mic": "Mostrar mapa de calor MFCC (entrada do microfone)",
        "mfcc_heatmap_title_mic": "Características MFCC (entrada do microfone)",
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
        "mic_start_info": "Clique no botão acima para começar a gravação.",
        "model_desc_rf": "Random Forest: Um método que toma a decisão final combinando muitas decisões simples",
        "model_desc_svm": "SVM: Um método que encontra a linha de fronteira para separar diferentes grupos de dados"
    },
    "vi": {
        "language_name": "Tiếng Việt (Vietnamese-베트남어)",
        "title": "Bộ Phân Loại Thể Loại Nhạc",
        "upload": "Tải lên một hoặc nhiều file .wav",
        "select_model": "Chọn mô hình",
        "download_rf": "⬇️ Tải xuống báo cáo phân loại Random Forest",
        "download_svm": "⬇️ Tải xuống báo cáo phân loại SVM",
        "show_heatmap_mic": "Hiển thị bản đồ nhiệt MFCC (đầu vào micrô)",
        "mfcc_heatmap_title_mic": "Đặc trưng MFCC (đầu vào micrô)",
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
        "mic_start_info": "Nhấp vào nút ở trên để bắt đầu ghi âm.",
        "model_desc_rf": "Rừng ngẫu nhiên: Phương pháp đưa ra quyết định cuối cùng bằng cách kết hợp nhiều quyết định đơn giản",
        "model_desc_svm": "SVM: Phương pháp tìm đường biên để phân tách các nhóm dữ liệu khác nhau"
    },
    "tr": {
        "language_name": "Türkçe (Turkish-튀르키에어)",
        "title": "Müzik Türü Sınıflandırıcı",
        "upload": "Bir veya daha fazla .wav dosyası yükleyin",
        "select_model": "Bir model seçin",
        "download_rf": "⬇️ Random Forest Sınıflandırma Raporunu İndir",
        "download_svm": "⬇️ SVM Sınıflandırma Raporunu İndir",
        "show_heatmap_mic": "MFCC Isı Haritasını Göster (Mikrofon Girişi)",
        "mfcc_heatmap_title_mic": "MFCC Özellikleri (Mikrofon Girişi)",
        "predicted_genre": "Tahmin Edilen Tür",
        "show_heatmap": "MFCC Isı Haritasını",
        "mic_start_info": "Kayda başlamak için yukarıdaki düğmeye tıklayın.",
        "model_desc_rf": "Random Forest: Birçok basit kararı birleştirerek nihai kararı veren yöntem",
        "model_desc_svm": "SVM: Farklı veri gruplarını ayıran sınır çizgisini bulan yöntem"
    }
}

# 언어 이름 리스트 생성 (Create list of language names)
language_names = [lang_dict[code]["language_name"] for code in lang_dict.keys()]

# 기본 언어 인덱스 (Default language index, e.g. English)
default_index = language_names.index("English (영어)")

# 선택된 언어 이름 받기 (Get selected language name from sidebar)
selected_language_name = st.sidebar.selectbox(
    "Choose Language / 언어 선택",
    options=language_names,
    index=default_index
)

# 선택된 언어명 바로 아래에 표시 (Display selected language name for debugging)
st.sidebar.write(f"Selected language: {selected_language_name}")

# 언어 이름 → 코드 역매핑 (Map selected language name back to language code)
language_code = list(lang_dict.keys())[language_names.index(selected_language_name)]

# texts 할당 (Assign texts dictionary for selected language)
texts = lang_dict[language_code]

# 페이지 설정 (Page setup)
st.set_page_config(page_title=texts["title"], layout="centered")

# 모델 선택 (Model selection)
model_option = st.radio(texts["select_model"], ("Random Forest", "SVM"))
model_file = "model.pkl" if model_option == "Random Forest" else "svm_model.pkl"
model = joblib.load(model_file)  # 모델 불러오기 (Load model)
scaler = joblib.load("scaler.pkl")  # 스케일러 불러오기 (Load scaler)
label_encoder = joblib.load("label_encoder.pkl")  # 레이블 인코더 불러오기 (Load label encoder)

# 모델 설명 안내문구 표시 (Show model description)
if model_option == "Random Forest":
    st.info(texts["model_desc_rf"])
elif model_option == "SVM":
    st.info(texts["model_desc_svm"])

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
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=29)  # n_mfcc= 13 → 29로 변경
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

# --- 실시간 마이크 녹음 기능 제거됨 (Streamlit Cloud 호환 안됨) ---
st.markdown("## 🎤 Real-Time Mic Recording")
st.info("🔇 Real-time mic recording is not supported in this version. Please upload a .wav file instead.")

# audio_bytes = audio_recorder() 등 삭제됨
# audio_bytes 사용도 삭제됨

# 🔽 기존 코드는 주석 처리 (원할 때 다시 살릴 수 있도록)
# from streamlit_audio_recorder import audio_recorder
# audio_bytes = audio_recorder()

# if audio_bytes:
#     st.audio(audio_bytes, format="audio/wav")
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
#             tmpfile.write(audio_bytes)
#             tmpfile_path = tmpfile.name

#         y, sr = librosa.load(tmpfile_path)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=29)
#         mfcc_mean = np.mean(mfcc, axis=1)
#         mfcc_std = np.std(mfcc, axis=1)
#         features = np.concatenate((mfcc_mean, mfcc_std)).reshape(1, -1)
#         features_scaled = scaler.transform(features)

#         prediction_encoded = model.predict(features_scaled)
#         prediction = label_encoder.inverse_transform(prediction_encoded)[0]
#         st.success(f"🎶 {texts['predicted_genre']} (Mic): `{prediction.capitalize()}`")

#         if hasattr(model, "predict_proba"):
#             proba = model.predict_proba(features_scaled)[0]
#             classes = label_encoder.inverse_transform(model.classes_)
#             proba_dict = dict(zip(classes, proba))
#             st.bar_chart(proba_dict)

#         if st.checkbox(texts["show_heatmap_mic"]):
#             fig, ax = plt.subplots(figsize=(8, 4))
#             sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
#             ax.set_title(texts["mfcc_heatmap_title_mic"])
#             ax.set_xlabel("Time")
#             ax.set_ylabel("MFCC Coefficients")
#             st.pyplot(fig)

#     except Exception as e:
#         st.error(f"마이크 오디오 처리 중 오류 발생: {e}")
# else:
#     st.info(texts["mic_start_info"])
