from rocket_fun.minirocket import fit, transform
import numpy as np
import pandas as pd
import librosa
import os
from joblib import load



def transform_sig(filepath):
     
    max_duration = 7.5
    freq = 16000
    samp_size = int(max_duration*freq)

    y, sr = librosa.load(filepath, sr=freq)
    if y.size > samp_size:
        y = y[:samp_size]
    elif y.size < samp_size:
        y = np.pad(y, (0, samp_size - len(y)))

    raw_sig = y
    parameters = fit(raw_sig)

    ext_features = transform(raw_sig, parameters)

    return ext_features

with open(r"rocket_cls\classifier0.pkl", "rb") as f:
    clf = load(f)

ext_features = transform_sig("placeholder_path")

clf.predict(ext_features)