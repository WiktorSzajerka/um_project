import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit

N_MFCC = 23
N_FFT = 1024
HOP_LENGTH = 250

AGE_GROUPS = ["teens", "twenties", "thirties", "forties", "fifties", "sixties"]

AUDIO_DIR = "cv-corpus/pl/clips"
METADATA_FILE = "cv-corpus/pl/validated.tsv"
MAX_PER_CLASS = 1000

def extract_mfcc(file_path, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)

        features = np.concatenate([
            np.mean(mfcc, axis=1),        # 23 cechy
            np.mean(delta_mfcc, axis=1),  # 23 cechy
            np.mean(delta2_mfcc, axis=1), # 23 cechy
            np.mean(rms, axis=1)          # 1 cecha
        ])
        return features

    except Exception as e:
        print(f"Błąd {file_path}: {e}")
        return None


def load_dataset(audio_dir, metadata_file, age_groups=AGE_GROUPS, max_per_class=1000):
    df = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    df = df[df["age"].notna()]
    df = df[df["age"].isin(age_groups)]

    samples = []
    for age in age_groups:
        group = df[df["age"] == age]
        samples.append(group.sample(min(len(group), max_per_class), random_state=42))
    df_balanced = pd.concat(samples).reset_index(drop=True)

    print(f"Rozkład klas po balansowaniu:\n{df_balanced['age'].value_counts()}\n")

    X, y = [], []
    total = len(df_balanced)

    for i, row in df_balanced.iterrows():
        file_path = os.path.join(audio_dir, row["path"])
        features = extract_mfcc(file_path)
        if features is not None:
            X.append(features)
            y.append(row["age"])

        if (i + 1) % 100 == 0:
            print(f"Postęp: {i + 1}/{total} plików")

    return np.array(X), np.array(y)


def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def split_data(X, y, test_size=0.2, random_state=42):
    splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train, test = next(splitter.split(X, y))
    return train, test