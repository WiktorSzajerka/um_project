import numpy as np
import pandas as pd
from aeon.classification.convolution_based import RocketClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from joblib import dump

sig_data = np.load("raw_signal.npy")
metadata = pd.read_csv("targets.csv")
targets = metadata["age"]

X_train, X_test, y_train, y_test = train_test_split(sig_data, targets, train_size=0.75, random_state=55, stratify=targets)
y_test.to_csv("targets_test.csv")

clf = RocketClassifier(class_weight="balanced", n_jobs=12)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"{balanced_accuracy_score(y_test, y_pred):.3f}")

with open("rocket1.pkl", "wb") as f:
    dump(clf, f, protocol=5)

