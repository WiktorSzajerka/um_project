import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


age_data = pd.read_csv("reduced_data.csv")
n_clips = age_data.shape[0]
targets = pd.DataFrame(columns=["path", "age", "gender"])
max_duration = 7.5
freq = 16000
max_size = int(max_duration*freq)
data = np.zeros((n_clips, max_size))
i = 0

for filename in os.listdir(r'pl\clips'):
    filepath = os.path.join(r'pl\clips', filename)
    if os.path.isfile(filepath):
        mask = age_data["path"] == filename
        if mask.any():
            y, sr = librosa.load(filepath, sr=freq)
            if y.size > max_size:
                y = y[:max_size]
            elif y.size < max_size:
                y = np.pad(y, (0, max_size - len(y)))

            targets.loc[i] = age_data[mask].iloc[0]
            data[i] = y
            i += 1
            if i%500 == 0:
                print(i, " done")
   

            
        


np.save("raw_signal.npy", data)
targets.to_csv("targets.csv")
