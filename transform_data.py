from rocket_fun.minirocket import fit, transform
import numpy as np
import pandas as pd
import librosa
import os



def transform_test(data_path, batch_size):
    age_data = pd.read_csv(data_path, sep="\t")
    age_data = age_data[["path", "age"]]
    age_data.loc[age_data["age"].isin(["sixties", "seventies", 'eighties', "nineties"]), "age"] = "sixties+"
    age_data.dropna(inplace=True)
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



X_test, _ = transform_test(data_path=rf"pl\validated.tsv", batch_size=5000)

np.save("transformed_rckt.npy", X_test)


