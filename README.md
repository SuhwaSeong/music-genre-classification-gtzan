````md
# Music Genre Classifier (Multilingual)

## About

This is a Streamlit web app that classifies music genres from audio (.wav) files.  
It supports three machine learning models: Random Forest, Support Vector Machine (SVM), and Convolutional Neural Network (CNN).  
The app also features automatic translation for a multilingual user interface using Google Translate.

## Features

- Upload `.wav` audio files for music genre classification  
- Choose from three models: Random Forest, SVM, or CNN  
- Visualize MFCC heatmaps and Mel spectrograms  
- Multilingual UI with automatic text translation (English, Korean, German, French, Spanish)  

## Installation and Running

### 1. Clone the repository

```bash
git clone https://github.com/SuhwaSeong/music-genre-classification-gtzan.git
cd music-genre-classification-gtzan
````

### 2. Install dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

### 3. Run the app

Start the Streamlit app by running:

```bash
streamlit run app.py
```

This will open the app in your default web browser.

## How to Use

1. Select your preferred language from the sidebar.
2. Upload a `.wav` music file.
3. Choose one of the three models (Random Forest, SVM, CNN).
4. The app will predict and display the music genre.
5. Optionally, visualize MFCC heatmaps or Mel spectrograms by checking the boxes.

## Notes

* On first run, the app will automatically download the CNN model and other required files.
* The app is optimized to run on Python 3.10.
* Translation uses the free Google Translate API via the `googletrans` library; translation accuracy may vary.

## Supported Languages

* English
* Korean
* German
* French
* Spanish

## Contact & Contribution

If you have questions or feature requests, please open an issue on GitHub.
Pull requests for improvements are always welcome!

**Enjoy classifying your music! 🎵**

```

필요하면 더 도와드릴 부분 있으면 알려주세요!
```
