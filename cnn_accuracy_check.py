import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 모델 경로
model_path = r"C:\Users\choho\OneDrive\바탕 화면\python practice\music-genre-classification-gtzan\cnn_genre_model.keras"

# 테스트 데이터셋 폴더 경로
test_folder = r"C:\Users\choho\OneDrive\바탕 화면\python practice\music-genre-classification-gtzan\test"

# 모델 불러오기
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터 불러오기 함수
def load_test_data(data_dir, max_len=128):
    X = []
    y = []
    genres = os.listdir(data_dir)
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        for file in os.listdir(genre_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_dir, file)
                try:
                    audio, sr = librosa.load(file_path, sr=22050)
                except Exception as e:
                    print(f"⚠️ 파일 무시됨: {file_path} ({e})")
                    continue
                mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                pad_width = max(0, max_len - mel_db.shape[1])
                mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant') if pad_width > 0 else mel_db[:, :max_len]
                X.append(mel_db[np.newaxis, ..., np.newaxis])
                y.append(genre)
    return np.vstack(X), np.array(y)

# 데이터 로딩
X_test, y_test_labels = load_test_data(test_folder)

# 라벨 인코딩
label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test_labels)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=10)

# 예측
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# 정확도 및 리포트
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("cnn_classification_report.csv", index=True)

# 결과 출력
print(f"CNN Accuracy: {report['accuracy']:.4f}")

# confusion matrix용 예측값 저장
pd.DataFrame(y_test).to_csv("cnn_y_test.csv", index=False)
pd.DataFrame(y_pred).to_csv("cnn_y_pred.csv", index=False)

# label encoder도 저장
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)