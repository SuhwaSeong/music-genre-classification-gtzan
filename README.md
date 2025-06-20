# 🎵 Music Genre Classification with GTZAN

A multilingual music genre classification web app built with Streamlit. It classifies `.wav` audio files into genres using machine learning models including CNN, Random Forest, and SVM. The app supports 18 languages and offers intuitive audio upload, real-time recording, and visual outputs such as MFCC heatmaps and prediction probabilities.

## 📂 Dataset

- **GTZAN Genre Collection**  
  10 genres × 100 samples (30-second `.wav` files)  
  Source: [Kaggle - GTZAN Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

## 🧠 Models Used

| Model           | Description                                     | Accuracy |
|----------------|--------------------------------------------------|----------|
| 🎲 Random Forest | MFCC features + classic ensemble learning       | ~64%     |
| 🔍 SVM           | Support Vector Machine with MFCC input          | ~61%     |
| 🧠 CNN (NEW)     | Deep learning model using mel spectrogram images | TBD      |

> CNN model is under training and will soon be integrated into the app.

## 🚀 Features

- 🎧 Upload or record `.wav` files
- 🌍 Multilingual UI (18 languages)
- 🔥 Visual MFCC heatmaps
- 📈 Display prediction probabilities
- 📦 Downloadable classification reports

## 🛠️ Tech Stack

- Python, Streamlit, NumPy, Librosa, Scikit-learn, TensorFlow/Keras
- Matplotlib, Seaborn, Pandas
- Streamlit-webrtc (for real-time recording)

## 🖥️ Demo

Try the live app 👉  
[https://music-genre-classification-gtzan-kbaft4cdqz6hd69hxkuwas.streamlit.app/](https://music-genre-classification-gtzan-kbaft4cdqz6hd69hxkuwas.streamlit.app/)

## 📌 To Do

- [x] Add multilingual support  
- [x] Add MFCC + heatmap display  
- [x] Add model selection (RF, SVM)  
- [ ] Train & integrate CNN  
- [ ] Improve performance with data augmentation  
- [ ] Add user feedback form for continuous learning  

## 📄 License

MIT License  
Feel free to use, modify, and contribute.

---
