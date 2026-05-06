from joblib import load

from mfcc_utils import extract_mfcc

AUDIO_PATH = "C:/Users/ester/Desktop/studia/uczenie maszynowe/projekt/projekt/cv-corpus/pl/clips/common_voice_pl_20554744.mp3"

MODELS = {
    "1": "nearest_centroid.pkl",
    "2": "random_forest.pkl",
    "3": "svc.pkl"
}

if __name__ == "__main__":
    print("Wybierz model:")
    print("1 - Nearest Centroid")
    print("2 - Random Forest")
    print("3 - SVC")
    choice = input("Wpisz numer: ")

    match choice:
        case "1" | "2" | "3":
            clf, scaler, le = load(MODELS[choice])
        case _:
            print("Nieprawidłowy wybór.")
            exit()

    features = extract_mfcc(AUDIO_PATH)
    X = scaler.transform(features.reshape(1, -1))
    y_pred = clf.predict(X)
    print(f"Przewidywana grupa wiekowa: {le.inverse_transform(y_pred)[0]}")