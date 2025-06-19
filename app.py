import streamlit as st
import librosa
import numpy as np
import joblib

st.title("🎧 Music Genre Classifier")
st.write("Upload a music file (.wav only) and I'll guess the genre!")

model = joblib.load("model.pkl")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.concatenate((mfcc_mean, mfcc_std)).reshape(1, -1)

    prediction = model.predict(features)[0]
    st.success(f"🎵 Predicted Genre: **{prediction.capitalize()}**")
