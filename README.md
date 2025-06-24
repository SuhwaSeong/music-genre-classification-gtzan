# 🎵 Music Genre Classifier (with CNN)

A multilingual web application built with Streamlit to classify music genres from `.wav` audio files using machine learning models.

---

## 🌟 Features | 주요 기능

### English
- 🌍 **Multilingual UI**: English, Korean, German, French, Spanish (auto-translated)
- 🎯 **Model Selection**: Random Forest, SVM, and CNN
- 🔍 **Feature Extraction**: MFCC (for RF/SVM) and Mel Spectrogram (for CNN)
- 📈 **Model Accuracy Chart**: Visual comparison of model accuracy
- 🎧 **Audio Playback**: Play uploaded audio in the browser
- 📊 **Prediction Probabilities**: View probabilities for all genres
- 🧩 **Confusion Matrix**: Visualized confusion matrix (CNN only, test data required)
- 🖼 **Waveform / Heatmap Visualization**: MFCC & Mel Spectrogram
- 💾 **Downloadable Results**: Save predictions as CSV
- 📝 **User Logs**: App automatically logs predictions and usage

### 한글
- 🌍 **다국어 UI 지원**: 영어, 한국어, 독일어, 프랑스어, 스페인어 (자동 번역)
- 🎯 **모델 선택 가능**: Random Forest, SVM, CNN
- 🔍 **특징 추출**: RF/SVM은 MFCC, CNN은 멜 스펙트로그램 사용
- 📈 **모델 정확도 시각화**: 바 차트로 정확도 비교
- 🎧 **오디오 재생 기능**: 업로드한 오디오 웹에서 바로 듣기
- 📊 **예측 확률 표시**: 각 장르별 예측 확률 시각화
- 🧩 **혼동 행렬**: CNN 예측 결과 평가 (테스트 데이터 필요)
- 🖼 **파형 / 히트맵 시각화**: 오디오 파형 및 MFCC/Mel 시각화
- 💾 **예측 결과 다운로드**: CSV 파일로 저장
- 📝 **사용자 로그 저장**: 예측 및 사용 정보 자동 기록

---

## 📂 Required Files (auto-downloaded from Google Drive)

These files will be downloaded automatically by the app:

| File                         | Purpose             |
|------------------------------|---------------------|
| `rf_model.pkl`               | Random Forest model |
| `svm_model.pkl`              | SVM model           |
| `cnn_genre_model.keras`      | CNN model           |
| `scaler.pkl`                 | Feature scaler      |
| `label_encoder.pkl`          | Label encoder       |
| `rf_classification_report.csv` | RF evaluation report |
| `svm_classification_report.csv` | SVM evaluation report |
| `cnn_classification_report.csv` | CNN evaluation report |
| `cnn_y_test.csv`             | CNN test labels     |
| `cnn_y_pred.csv`             | CNN predicted labels|

> ⚠️ Make sure all files are publicly shared via Google Drive.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/music-genre-classifier.git
   cd music-genre-classifier

2. Install requirements:

pip install -r requirements.txt


3. Launch the Streamlit app:

streamlit run app.py




---

🧪 Model Information

Random Forest & SVM:

Features: 13 MFCCs + stats

Accuracy: ~64% (RF), ~61% (SVM)


CNN:

Input: Mel Spectrogram (128×128×1)

Accuracy: ~70% (test accuracy on GTZAN subset)




---

📊 Dataset

Used GTZAN music genre dataset (10 genres, 1000 samples):

blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

CNN input generated using Librosa Mel Spectrograms



---

📄 License

This project is under the MIT License.
Feel free to modify and use it for educational purposes.


---

🙋‍♀️ Author

Suhwa Seong
🎓 Master’s in Data Science | 🎤 Background in Vocal Music
📍 Brandenburg an der Havel, Germany
🔗 LinkedIn | GitHub


---

---

필요한 경우 다음도 함께 만들어드릴 수 있습니다:

- `requirements.txt` (라이브러리 목록)
- `.streamlit/config.toml` (Python 버전 고정용)
- `.gitattributes` or `.gitignore` 파일

GitHub에 업로드하신 후 확인 원하시면 링크를 주셔도 됩니다!

