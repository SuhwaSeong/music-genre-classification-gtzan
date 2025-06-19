# 🎧 Music Genre Classification with GTZAN Dataset

GTZAN-based audio genre classification using MFCC and Random Forest

---

## 📚 Dataset
- **GTZAN** dataset: 10 genres × 100 audio tracks (30 seconds each)
- Dataset includes: `genres_original/`, `images_original/`, and CSVs with precomputed features

---

## 💡 What This Project Does
- Extracts 13 MFCCs per audio file
- Computes mean and standard deviation (total 26 features)
- Builds a **Random Forest classifier** using scikit-learn
- Visualizes predictions with a **confusion matrix**
- Outputs a **classification report** (precision, recall, F1-score)

---

## 🔍 Result Summary

| Metric         | Value    |
|----------------|----------|
| Accuracy       | ~64%     |
| Best Genres    | Classical, Metal, Jazz |

- Classical and Metal genres were predicted with high precision and recall
- Rock and Country showed more frequent confusion with other genres

---

## 🛠 Tools & Libraries

- Python 3.x  
- Google Colab  
- `librosa`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 📁 File Structure

Here’s how the project files are organized:

music-genre-classification-gtzan/
│
├── GTZAN_MFCC_Classification.ipynb ← the main Colab notebook
├── requirements.txt ← list of Python libraries used
├── README.md ← this file
└── images/ ← (optional) plots like confusion matrix


---

## 🚀 What I Want to Do Next

- Try other features like Chroma or Spectral Contrast
- Test different models like SVM or deep learning
- Turn this into a small app using Streamlit so anyone can try genre prediction
- Use 3-second clips to make the dataset bigger

---

## 👩‍💻 About Me

I’m **Suhwa Seong**, a master's student in Data Science with a background in music.  
I love projects that combine data and creativity.

---

## 📜 License

No license yet — I’ll add one later (MIT or open-source).
