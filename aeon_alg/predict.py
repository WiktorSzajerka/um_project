import numpy as np
import librosa
from joblib import load

with open(r"aeon_alg\rocket1.pkl", "rb") as f:
    clf = load(f)

filepath = r"pl\clips\common_voice_pl_20547774.mp3"
max_duration = 7.5
freq = 16000
max_size = int(max_duration*freq)

y, sr = librosa.load(filepath, sr=freq)
if y.size > max_size:
    y = y[:max_size]
elif y.size < max_size:
    y = np.pad(y, (0, max_size - len(y)))


y_pred = clf.predict(np.array([y]))
print(y_pred)