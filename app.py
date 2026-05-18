import tempfile
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from joblib import load

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from mfcc_algs.mfcc_utils import extract_mfcc
from rocket_fun.minirocket import fit, transform


MAX_DURATION_SECONDS = 7.5
RAW_SAMPLE_RATE = 16_000
RAW_SIGNAL_SIZE = int(MAX_DURATION_SECONDS * RAW_SAMPLE_RATE)


MFCC_MODEL_NAMES = {
    "nearest_centroid",
    "random_forest",
    "svc",
}


LABELS = {
    "teens": "10-19",
    "twenties": "20-29",
    "thirties": "30-39",
    "fourties": "40-49",
    "fifties": "50-59",
    "sixties": "60-69",
    "sixties+": "60+",
    "seventies": "70-79",
    "eighties": "80-89",
    "nineties": "90-99",
}


def find_models() -> list[Path]:
    project_path = Path(__file__).resolve().parent

    ignored_dirs = {
        ".git",
        ".idea",
        ".venv",
        "venv",
        "__pycache__",
    }

    paths: list[Path] = []

    for path in project_path.rglob("*.pkl"):
        if any(part in ignored_dirs for part in path.parts):
            continue

        paths.append(path)

    return sorted(paths)


def check_model_type(path: Path) -> str:
    model_name = path.stem.lower()

    if model_name in MFCC_MODEL_NAMES:
        return "mfcc"

    if "rocket_cls" in path.parts:
        return "rocket"

    return "raw"


def display_model_name(path: Path) -> str:
    names = {
        "nearest_centroid": "Nearest Centroid",
        "random_forest": "Random Forest",
        "svc": "SVC",
        "rocket1": "ROCKET",
        "classifier0": "ROCKET",
        "clssifier0": "ROCKET (classifier0)",
    }

    model_name = path.stem.lower()

    if model_name in names:
        return names[model_name]

    return path.name


def raw_signal(audio_path: str) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=RAW_SAMPLE_RATE, mono=True)
    y = y.astype(np.float32)

    if y.size > RAW_SIGNAL_SIZE:
        y = y[:RAW_SIGNAL_SIZE]
    elif y.size < RAW_SIGNAL_SIZE:
        y = np.pad(y, (0, RAW_SIGNAL_SIZE - y.size))

    return np.array([y])


def rocket_features(audio_path: str) -> np.ndarray:
    raw_sig = raw_signal(audio_path)

    parameters = fit(raw_sig)
    ext_features = transform(raw_sig, parameters)

    return ext_features


def mfcc_features(audio_path: str) -> np.ndarray:
    features = extract_mfcc(audio_path)

    if features is None:
        raise RuntimeError("Nie udało się wyekstrahować cech MFCC z pliku audio.")

    return features.reshape(1, -1)


def label_to_text(label: Any) -> str:
    text = str(label)
    hint = LABELS.get(text)

    if hint is not None:
        return f"{text} ({hint})"

    return text


def predict_mfcc(model, audio_path: str) -> tuple[str, str]:
    clf, scaler, le = model[:3]

    features = mfcc_features(audio_path)
    x = scaler.transform(features)

    y_pred = clf.predict(x)
    predicted_label = le.inverse_transform(y_pred)[0]

    return label_to_text(predicted_label), "—"


def predict_rocket(model, audio_path: str) -> tuple[str, str]:
    if not hasattr(model, "predict"):
        raise RuntimeError("Ten plik .pkl nie zawiera obiektu z metodą predict().")

    x = rocket_features(audio_path)

    y_pred = model.predict(x)
    predicted_label = y_pred[0]

    predicted_array = np.asarray(predicted_label)

    if predicted_array.ndim > 0 and predicted_array.size == 1:
        predicted_label = predicted_array.item()

    return label_to_text(predicted_label), "—"


def predict_raw(model, audio_path: str) -> tuple[str, str]:
    if not hasattr(model, "predict"):
        raise RuntimeError("Ten plik .pkl nie zawiera obiektu z metodą predict().")

    x = raw_signal(audio_path)

    y_pred = model.predict(x)
    predicted_label = y_pred[0]

    predicted_array = np.asarray(predicted_label)

    if predicted_array.ndim > 0 and predicted_array.size == 1:
        predicted_label = predicted_array.item()

    return label_to_text(predicted_label), "—"


def predict_age_for_file(audio_path: str) -> list[list[str]]:
    model_paths = find_models()

    if not model_paths:
        raise RuntimeError("Nie znaleziono żadnego pliku .pkl w projekcie.")

    rows: list[list[str]] = []

    for path in model_paths:
        kind = check_model_type(path)
        model_name = display_model_name(path)

        try:
            model = load(path)

            if kind == "mfcc":
                predicted_age, note = predict_mfcc(model, audio_path)
                input_type = "MFCC features"

            elif kind == "rocket":
                predicted_age, note = predict_rocket(model, audio_path)
                input_type = "ROCKET features"

            else:
                predicted_age, note = predict_raw(model, audio_path)
                input_type = "raw audio"

        except Exception as exc:
            predicted_age = "—"
            note = f"Błąd: {exc}"
            input_type = kind

        rows.append(
            [
                model_name,
                input_type,
                predicted_age,
                note,
            ]
        )

    return rows


class AgeEstimatorDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Estymacja wieku na podstawie głosu")
        self.root.geometry("950x560")
        self.root.minsize(850, 500)

        self.selected_audio_path: str | None = None
        self.temp_dir = tempfile.TemporaryDirectory()

        self.create_widgets()
        self.refresh_models()

    def create_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.LabelFrame(main_frame, text="Dane wejściowe", padding=12)
        controls_frame.pack(fill=tk.X)

        self.audio_path_var = tk.StringVar(value="Nie wybrano pliku audio.")

        audio_path_label = ttk.Label(
            controls_frame,
            textvariable=self.audio_path_var,
            wraplength=850,
        )
        audio_path_label.grid(row=0, column=0, columnspan=6, sticky=tk.W, pady=(0, 10))

        choose_button = ttk.Button(
            controls_frame,
            text="Wybierz plik audio",
            command=self.choose_audio_file,
        )
        choose_button.grid(row=1, column=0, sticky=tk.W, padx=(0, 8))

        self.record_button = ttk.Button(
            controls_frame,
            text="Nagraj próbkę",
            command=self.start_recording,
        )
        self.record_button.grid(row=1, column=1, sticky=tk.W, padx=(8, 0))

        actions_frame = ttk.Frame(main_frame)
        actions_frame.pack(fill=tk.X, pady=14)

        self.predict_button = ttk.Button(
            actions_frame,
            text="Oszacuj wiek",
            command=self.start_prediction,
        )
        self.predict_button.pack(side=tk.LEFT)

        refresh_button = ttk.Button(
            actions_frame,
            text="Odśwież listę modeli",
            command=self.refresh_models,
        )
        refresh_button.pack(side=tk.LEFT, padx=(8, 0))

        self.models_info_var = tk.StringVar(value="")
        models_info_label = ttk.Label(actions_frame, textvariable=self.models_info_var)
        models_info_label.pack(side=tk.LEFT, padx=(16, 0))

        results_frame = ttk.LabelFrame(main_frame, text="Wyniki", padding=8)
        results_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("model", "input_type", "age_class", "notes")

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=columns,
            show="headings",
            height=12,
        )

        self.results_tree.heading("model", text="Model")
        self.results_tree.heading("input_type", text="Typ wejścia")
        self.results_tree.heading("age_class", text="Oszacowana klasa wieku")
        self.results_tree.heading("notes", text="Uwagi")

        self.results_tree.column("model", width=150)
        self.results_tree.column("input_type", width=100)
        self.results_tree.column("age_class", width=150)
        self.results_tree.column("notes", width=300)

        y_scrollbar = ttk.Scrollbar(
            results_frame,
            orient=tk.VERTICAL,
            command=self.results_tree.yview,
        )
        self.results_tree.configure(yscrollcommand=y_scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_var = tk.StringVar(value="Gotowe.")

        status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=4,
        )
        status_label.pack(fill=tk.X, pady=(10, 0))

    def choose_audio_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz plik audio",
            filetypes=[
                ("Pliki audio", "*.wav *.mp3 *.m4a"),
                ("Wszystkie pliki", "*.*"),
            ],
        )

        if not path:
            return

        self.selected_audio_path = path
        self.audio_path_var.set(path)
        self.status_var.set("Wybrano plik audio.")

    def start_recording(self) -> None:
        duration = MAX_DURATION_SECONDS

        self.record_button.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.DISABLED)

        self.status_var.set(f"Nagrywanie przez {duration:.1f} s...")

        thread = threading.Thread(
            target=self.record_audio,
            args=(duration,),
            daemon=True,
        )
        thread.start()

    def record_audio(self, duration: float) -> None:
        try:
            samples_count = int(duration * RAW_SAMPLE_RATE)

            recording = sd.rec(
                samples_count,
                samplerate=RAW_SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            output_path = Path(self.temp_dir.name) / "recorded_voice.wav"
            sf.write(output_path, recording, RAW_SAMPLE_RATE)

            self.root.after(
                0,
                lambda: self.finish_recording(str(output_path)),
            )

        except Exception as exc:
            self.root.after(
                0,
                lambda: self.show_error(f"Nie udało się nagrać audio: {exc}"),
            )

    def finish_recording(self, path: str) -> None:
        self.selected_audio_path = path
        self.audio_path_var.set(f"Nagrano próbkę: {path}")

        self.status_var.set("Nagrywanie zakończone.")

        self.record_button.config(state=tk.NORMAL)
        self.predict_button.config(state=tk.NORMAL)

    def start_prediction(self) -> None:
        if not self.selected_audio_path:
            messagebox.showwarning(
                "Brak pliku audio",
                "Najpierw wybierz plik audio albo nagraj próbkę głosu.",
            )
            return

        self.clear_results()

        self.predict_button.config(state=tk.DISABLED)
        self.record_button.config(state=tk.DISABLED)

        self.status_var.set("Trwa estymacja wieku...")

        thread = threading.Thread(
            target=self.predict_audio,
            args=(self.selected_audio_path,),
            daemon=True,
        )
        thread.start()

    def predict_audio(self, audio_path: str) -> None:
        try:
            rows = predict_age_for_file(audio_path)

            self.root.after(
                0,
                lambda: self.show_results(rows),
            )

        except Exception as exc:
            self.root.after(
                0,
                lambda: self.show_error(str(exc)),
            )

    def show_results(self, rows: list[list[str]]) -> None:
        self.clear_results()

        for row in rows:
            self.results_tree.insert("", tk.END, values=row)

        ok_count = sum(1 for row in rows if not str(row[-1]).startswith("Błąd:"))

        self.status_var.set(
            f"Poprawnie wykonano predykcję dla {ok_count}/{len(rows)} modeli."
        )

        self.predict_button.config(state=tk.NORMAL)
        self.record_button.config(state=tk.NORMAL)

    def show_error(self, message: str) -> None:
        self.status_var.set("Wystąpił błąd.")

        self.predict_button.config(state=tk.NORMAL)
        self.record_button.config(state=tk.NORMAL)

        messagebox.showerror("Błąd", message)

    def refresh_models(self) -> None:
        model_paths = find_models()

        self.models_info_var.set(f"Liczba znalezionych modeli: {len(model_paths)}")
        self.status_var.set("Odświeżono listę modeli.")

    def clear_results(self) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

    def close(self) -> None:
        self.temp_dir.cleanup()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()

    app = AgeEstimatorDesktopApp(root)

    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()