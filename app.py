import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from io import BytesIO
import tensorflow as tf

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
        "language_name": "हिन्दी (Hindi-인도-힌디어)",        
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
        "language_name": "Español (Spanish-스페인어)",
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
        "language_name": "Türkçe (Turkish-튀르키예어)",
        "title": "Müzik Türü Sınıflandırıcı",
        "upload": "Bir veya daha fazla .wav dosyası yükleyin",
        "select_model": "Bir model seçin",
        "download_rf": "⬇️ Random Forest Sınıflandırma Raporunu İndir",
        "download_svm": "⬇️ SVM Sınıflandırma Raporunu İndir",
        "show_heatmap_mic": "MFCC Isı Haritasını Göster (Mikrofon Girişi)",
        "mfcc_heatmap_title_mic": "MFCC Özellikleri (Mikrofon Girişi)",
        "predicted_genre": "Tahmin Edilen Tür",
        "show_heatmap": "MFCC Isı Haritasını",
        "accuracy_summary": "Model doğruluk özeti",
        "accuracy_rf": "Random Forest doğruluğu",
        "accuracy_svm": "SVM doğruluğu",
        "best_genres": "En iyi performans gösteren türler",
        "about_app": "Bu uygulama hakkında",
        "model_performance": "Model performans ölçütleri",
        "select_file": "Sınıflandırmak için bir dosya seçin",
        "choose_language": "Dil seçin / Choose Language / 언어 선택",
        "start_info": "Başlamak için bir veya daha fazla .wav dosyası yükleyin.",
        "mic_start_info": "Kayda başlamak için yukarıdaki düğmeye tıklayın.",
        "model_desc_rf": "Random Forest: Birçok basit kararı birleştirerek nihai kararı veren yöntem",
        "model_desc_svm": "SVM: Farklı veri gruplarını ayıran sınır çizgisini bulan yöntem"
    },
}

# --- 장르 레이블 (CNN용)
genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']

# --- 설정 상수 ---
BASE_PATH = ""
MODEL_FILES = {
    "Random Forest": "rf_model.pkl",
    "SVM": "svm_model.pkl",
    "CNN": "cnn_genre_model.keras"
}
REPORT_FILES = {
    "Random Forest": "rf_classification_report.csv",
    "SVM": "svm_classification_report.csv"
}
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
SAMPLE_AUDIO_FILE = "sample.wav"
N_MFCC = 13

# --- 유틸 함수 ---
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model(MODEL_FILES["CNN"], compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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

def extract_features(audio_bytes, n_mfcc):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.concatenate([mfcc_mean, mfcc_std]).reshape(1, -1)
    return features, mfcc

def extract_mel_spectrogram(audio_bytes, max_len=128):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] > max_len:
        mel_db = mel_db[:, :max_len]
    else:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mel_db[np.newaxis, ..., np.newaxis], mel_db

def check_class_alignment(model, label_encoder):
    try:
        model_classes = label_encoder.inverse_transform(model.classes_)
    except Exception:
        model_classes = label_encoder.classes_
    return model_classes

# --- 앱 시작 ---
st.set_page_config(page_title="Music Genre Classifier", layout="centered")
st.title("🎵 Music Genre Classifier (with CNN Support)")

model_option = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

if model_option == "CNN":
    model = load_cnn_model()
else:
    model, scaler, label_encoder, report_df, report_data, report_path = load_model_files(model_option)
    model_classes = check_class_alignment(model, label_encoder)

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

    if model_option == "CNN":
        features, mel = extract_mel_spectrogram(audio_bytes)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_label = genre_labels[predicted_index]

        st.success(f"🎶 Predicted Genre: `{predicted_label.capitalize()}`")

        st.markdown("### 🔍 Prediction Probabilities")
        proba_dict = dict(zip(genre_labels, prediction[0]))
        st.bar_chart(proba_dict)

        if st.checkbox("Show Mel Spectrogram"):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(mel, cmap="YlGnBu", ax=ax)
            ax.set_title("Mel Spectrogram")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mel Bands")
            st.pyplot(fig)
            plt.close(fig)

    else:
        try:
            features, mfcc = extract_features(audio_bytes, N_MFCC)
            features_scaled = scaler.transform(features)
            prediction_encoded = model.predict(features_scaled)
            prediction = label_encoder.inverse_transform(prediction_encoded)
            st.success(f"🎶 Predicted Genre: `{prediction[0].capitalize()}`")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_scaled)[0]
                proba_dict = dict(zip(model_classes, proba))
                st.markdown("### 🔍 Prediction Probabilities")
                st.bar_chart(proba_dict)

            if st.checkbox("Show MFCC Heatmap"):
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(mfcc, cmap="YlGnBu", ax=ax)
                ax.set_title("MFCC Features")
                ax.set_xlabel("Time")
                ax.set_ylabel("MFCC Coefficients")
                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.error("❌ Error during prediction.")
            st.exception(e)
else:
    st.info("Please upload a .wav file to get started.")


