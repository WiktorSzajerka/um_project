from mfcc_algs.mfcc_utils import extract_mfcc
import numpy as np
import pandas as pd
import librosa
import os
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

    age_data = pd.concat([df_sampled, df_upsampled]).reset_index(drop=True)
    return age_data

def transform_train(data_path):
    age_data = balance_classes(data_path)
    ext_features = np.zeros((age_data.shape[0], 70))
    for i, (_, row) in enumerate(age_data.iterrows()):
        filename = row["path"]
        filepath = os.path.join(r'pl\clips', filename)
        features = extract_mfcc(filepath)
        ext_features[i] = features

    targets = age_data["age"]
    print(f"100% done")
    
    return ext_features, targets

def transform_test(data_path):
    age_data = pd.read_csv(data_path)
    ext_features = np.zeros((age_data.shape[0], 70))
    for i, (_, row) in enumerate(age_data.iterrows()):
        filename = row["path"]
        filepath = os.path.join(r'pl\clips', filename)
        features = extract_mfcc(filepath)
        ext_features[i] = features

    targets = age_data["age"]
    print(f"100% done")
    
    return ext_features, targets

scaler = StandardScaler()


models = {
    "nearest_centroid": NearestCentroid(),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=12),
    "svc": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
}

classifiers_scores = [[],[],[]]

for j in range(1,10):
    for ind, (name, clf) in enumerate(models.items()):
        X_train, y_train = transform_train(data_path=rf"splits\fold{j}train.csv")
        X_test, y_test = transform_test(data_path=rf"splits\fold{j}test.csv")
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        clf = clone(clf)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=["teens", "twenties", "thirties", "fourties", "fifties", 'sixties+'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["teens", "twenties", "thirties", "fourties", "fifties", 'sixties+'])
        disp.plot()
        plt.savefig(f"images\Confusion matrix {name}{j}")
        plt.close()
        classifiers_scores[ind].append(score)
        with open(rf"other_cls\{name}{j}.pkl", 'wb') as fo:
            dump(clf, fo)
    

pd.DataFrame(classifiers_scores[0], columns=["nearest_centroid"]).to_csv("nearest_centroid_scores")

pd.DataFrame(classifiers_scores[1], columns=["random_forest"]).to_csv("random_forest_scores")

pd.DataFrame(classifiers_scores[1], columns=["svc"]).to_csv("svc_scores")