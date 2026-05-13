from rocket_fun.minirocket import fit, transform
import numpy as np
import pandas as pd
import librosa
import os
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import matplotlib.pyplot as plt
from joblib import dump


def transform_sig(data_path, batch_size):
    age_data = pd.read_csv(data_path)
    chunks = [age_data.iloc[i:i+batch_size] for i in range(0, len(age_data), batch_size)]
    max_duration = 7.5
    freq = 16000
    samp_size = int(max_duration*freq)
    ext_features_list = []

    for ind, chunk in enumerate(chunks):
        raw_sig = np.zeros((chunk.shape[0], samp_size), dtype=np.float32)
        if ind != 0:
            print(F"{((ind)*batch_size/age_data.shape[0]):.0%} done")
        for i, (_, row) in enumerate(chunk.iterrows()):
            filename = row["path"]
            filepath = os.path.join(r'pl\clips', filename)

            y, sr = librosa.load(filepath, sr=freq)
            if y.size > samp_size:
                y = y[:samp_size]
            elif y.size < samp_size:
                y = np.pad(y, (0, samp_size - len(y)))

            raw_sig[i] = y

        parameters = fit(raw_sig)

        ext_features = transform(raw_sig, parameters)
        ext_features_list.append(ext_features)

        
    ext_features = np.concatenate(ext_features_list, axis=0)
    targets = age_data["age"]
    print(f"100% done")
    
    return ext_features, targets


log_cls = LogisticRegression(class_weight="balanced", n_jobs=12)
classifiers_scores = []

for j in range(10):
    print(f"Transforming {j} fold train data")
    X_train, y_train = transform_sig(data_path=rf"splits\fold{j}train.csv", batch_size=5000)
    print(f"Transforming {j} fold test data")
    X_test, y_test = transform_sig(data_path=rf"splits\fold{j}train.csv", batch_size=5000)
    clf = clone(log_cls)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["teens", "twenties", "thirties", "fourties", "fifties", 'sixties+'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["teens", "twenties", "thirties", "fourties", "fifties", 'sixties+'])
    disp.plot()
    plt.savefig(f"Confusion matrix rocket{j}")
    plt.close()
    classifiers_scores.append(score)
    with open(rf"rocket_cls\logcls{j}.pkl", 'wb') as fo:
        dump(clf, fo)
    

pd.DataFrame(classifiers_scores, columns=["rocket"]).to_csv("rckt_cls_scores")

