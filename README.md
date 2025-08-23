# 🎵 Music Genre Classifier (Multilingual, CNN-supported)

A multilingual Streamlit web app to classify music genres from `.wav` files using Random Forest, SVM, and CNN models.

---

## 🌟 Features

- 🎯 Model Selection: Random Forest, SVM, CNN  
- 🌍 Multilingual UI: English, Korean, German, French, Spanish  
- 🎧 Audio Playback in-browser  
- 🔍 Feature Extraction:  
  - RF/SVM: MFCC (13 coefficients + mean/std)  
  - CNN: Mel Spectrogram (128×128×1)  
- 📊 Model Accuracy Comparison  
- 🧩 Confusion Matrix (CNN only)  
- 🖼 Waveform, MFCC, and Spectrogram Visualization  
- 💾 CSV Download of Prediction Results  
- 📝 User Logs (app usage saved in `user_logs.csv`)  
- ☁️ Auto-download model files from Google Drive  

---

## 📊 Model Accuracy (GTZAN subset)

- 🧠 CNN: **79.1%**  
- 🎲 Random Forest: **75.5%**  
- 📐 SVM: **19.5%**

---

## ⚠️ SVM Performance Analysis

The SVM model underperformed significantly compared to RF and CNN. This is likely due to:

- Limited input features (only 13 MFCCs with mean and standard deviation)  
- Absence of hyperparameter tuning (`C`, `gamma`, `kernel`)  
- Lack of nonlinear kernel support for capturing complex patterns  
- Possibly unbalanced train-test split (missing stratification)  

This poor result emphasized the importance of model selection and feature representation in audio classification tasks.

---

## 📁 Dataset

- **GTZAN Music Genre Dataset**  
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock  
- 1000 samples (100 per genre)

---

## 🧪 Model Details

| Model          | Input Features             | Accuracy |
|----------------|----------------------------|----------|
| Random Forest  | 13 MFCCs + stats           | 75.5%    |
| SVM            | 13 MFCCs + stats           | 19.5%    |
| CNN            | Mel Spectrogram (128x128)  | 79.1%    |

---

## 🚀 Run Locally

```bash
git clone https://github.com/SuhwaSeong/music-genre-classification-gtzan.git
cd music-genre-classification-gtzan
pip install -r requirements.txt
streamlit run app.py

````
## Problem & Solution (KR/EN)

**[KR 문제]** 데이터가 적은 오디오 분류에서 과적합을 줄이는 3가지 방법을 제시하라.  
**[KR 해설(요지)]** 데이터 증강(노이즈/타임시프트/피치 변경), 규제(Dropout/L2), 검증전략(Stratified K-Fold + Early Stopping).

**[EN Problem]** Propose three strategies to reduce overfitting in low-data audio classification.  
**[EN Solution (summary)]** Augmentation (noise/time-shift/pitch), Regularization (Dropout/L2), Validation (stratified k-fold + early stopping).

**Rubric (10)** Coverage(3) • Justification(3) • Practicality(2) • Clarity(2)

---

## 👩‍🎓 Author

**Suhwa Seong**
🎓 Master's in Data Science
🎤 Background in Vocal Music
📍 Brandenburg an der Havel, Germany
🔗 [LinkedIn](https://www.linkedin.com/in/suhwa-seong-366150361) | [GitHub](https://github.com/SuhwaSeong)

---

## 📄 License

This project is licensed under the MIT License.


