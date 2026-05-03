import numpy as np
from joblib import load

with open("rocket1.pkl", "rb") as f:
    clf = load(f)


data = np.load("raw_signal.npy")
print(data[1])
y_pred = clf.predict(data[1:2,:])
print(y_pred)