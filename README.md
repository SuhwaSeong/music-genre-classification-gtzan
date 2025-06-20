# Music Genre Classifier

🎵 **Music Genre Classifier** — A multilingual audio classification web app built with Streamlit

---

## Overview

This project classifies music genres from `.wav` audio files using machine learning models (Random Forest and SVM) in Python.
It extracts MFCC (Mel Frequency Cepstral Coefficients) features from audio and inputs them into the models, providing classification results along with visualizations.

---

## Features

* Multilingual user interface supporting 20+ languages (including Korean, English, German)
* Choice between Random Forest and SVM classification models
* Upload multiple `.wav` files and select files for individual classification
* Displays predicted genre with probability bar charts
* Visualizes MFCC heatmaps for audio features
* Download classification reports and sample audio files

---

## Installation & Usage

1. Install Python 3.8 or higher
2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```
3. Set the path of model files (`rf_model.pkl`, `svm_model.pkl`, etc.) and report CSV files in the `BASE_PATH` variable inside `app.py`.
4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## How to Use

1. Select your preferred language in the sidebar.
2. Choose the classification model (Random Forest or SVM).
3. Upload one or more `.wav` audio files.
4. Select a file from the uploaded list to classify.
5. View the predicted genre, prediction probabilities, and MFCC heatmap visualization.
6. Download the classification report and sample audio via sidebar buttons.

---

## Contributing

Contributions are welcome! If you are interested or have suggestions:

* Please open issues on the GitHub repository for bugs or feature requests.
* Submit pull requests (PRs) with your changes for new features or bug fixes.
* Include detailed descriptions of your changes in the PR to help with review.
* Please follow the existing code style and conventions.

---

## Reporting Issues & Support

* If you encounter any problems or have ideas for improvement, please report them via GitHub issues.
* For urgent inquiries, feel free to contact me at [suhwa.seong86@gmail.com](mailto:suhwa.seong86@gmail.com).
* I will do my best to respond promptly.

---

## Developer

**Suhwa Seong**

* Email: [suhwa.seong86@gmail.com](mailto:suhwa.seong86@gmail.com)
* GitHub: [https://github.com/SuhwaSeong/music-genre-classification-gtzan](https://github.com/SuhwaSeong/music-genre-classification-gtzan)

---

## License

This project is licensed under the MIT License.

---
