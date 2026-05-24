from rocket_fun.minirocket import fit, transform
import numpy as np
import pandas as pd
import librosa
import os
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import RidgeClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt
from joblib import dump



def balance_classes(data_path):
    age_groups = ["teens", "twenties", "thirties", "fourties"]
    age_data = pd.read_csv(data_path)
    min_cl = age_data["age"].value_counts().min()

    mask = age_data["age"].isin(age_groups)
    df_to_reduce = age_data[mask]
    df_to_upsample = age_data[~mask]

    df_sampled = (
        df_to_reduce
        .groupby("age")[["path","age"]]
        .apply(lambda x: x.sample(n=min(min_cl*2, len(x))))
    )

    df_upsampled = (
        df_to_upsample
        .groupby("age", group_keys=False)
        .apply(lambda x: x.sample(n=min_cl*2, replace=True))
    )

    age_data = pd.concat([df_sampled, df_upsampled])
    return age_data

def transform_train(data_path, batch_size):
    age_data = balance_classes(data_path)
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

data = np.load("transformed_rckt.npy")


classifiers = []
alphas = np.logspace(-3,1,100)
for a in alphas:
    classifiers.append(RidgeClassifier(class_weight="balanced", alpha=a))
classifiers_scores = []


print(f"Transforming fold train data")

X_train , y_train = transform_train(r"splits\fold0train.csv", batch_size=3000)

print(f"Transforming fold test data")
test_targets = pd.read_csv(r"splits\fold0test.csv")
X_test = data[test_targets.index]
y_test = test_targets["age"]

for clf in classifiers:
    clf = clone(clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    classifiers_scores.append(score)

print(f"Transforming 2 fold train data")

X_train , y_train = transform_train(r"splits\fold1train.csv", batch_size=3000)

print(f"Transforming 2 fold test data")
test_targets = pd.read_csv(r"splits\fold1test.csv")
X_test = data[test_targets.index]
y_test = test_targets["age"]


for i, clf in enumerate(classifiers):
    clf = clone(clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    classifiers_scores[i] = (classifiers_scores[i]+score)/2

best = max(classifiers_scores)
print(alphas[classifiers_scores.index(best)])

pd.DataFrame(classifiers_scores, columns=["rocket"]).to_csv("fine_tuning_scores")

