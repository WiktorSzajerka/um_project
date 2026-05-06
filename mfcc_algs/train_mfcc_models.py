from joblib import dump
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from mfcc_utils import load_dataset, encode_labels, split_data, AGE_GROUPS, AUDIO_DIR, METADATA_FILE, MAX_PER_CLASS

if __name__ == "__main__":

    X, y_raw = load_dataset(AUDIO_DIR, METADATA_FILE, age_groups=AGE_GROUPS, max_per_class=MAX_PER_CLASS)

    y, le = encode_labels(y_raw)

    train, test = split_data(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train])
    X_test  = scaler.transform(X[test])

    models = {
        "nearest_centroid": NearestCentroid(),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "svc": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
    }

    for name, clf in models.items():
        print(f"=== Trenowanie {name} ===")
        clf.fit(X_train, y[train])
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y[test], y_pred)
        bal_acc = balanced_accuracy_score(y[test], y_pred)

        print(f"accuracy:          {acc:.5f}")
        print(f"balanced_accuracy: {bal_acc:.5f}\n")

        dump((clf, scaler, le), f"{name}.pkl")
        print(f"Model zapisany: {name}.pkl\n")