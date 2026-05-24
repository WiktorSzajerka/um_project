from mfcc_algs.mfcc_utils import extract_mfcc
import numpy as np
import pandas as pd
import librosa
import os

def transform_test(data_path):
    age_data = pd.read_csv(data_path, sep="\t")
    age_data = age_data[["path", "age"]]
    age_data.loc[age_data["age"].isin(["sixties", "seventies", 'eighties', "nineties"]), "age"] = "sixties+"
    age_data.dropna(inplace=True)
    ext_features = np.zeros((age_data.shape[0], 70))
    for i, (_, row) in enumerate(age_data.iterrows()):
        filename = row["path"]
        filepath = os.path.join(r'pl\clips', filename)
        features = extract_mfcc(filepath)
        ext_features[i] = features
        if i % 5000 == 0:
            print(F"{(i/age_data.shape[0]):.0%} done")
    print(f"100% done")
    
    return ext_features

extracted_feat = transform_test(r"pl\validated.tsv")

np.save("mfcc_feat.npy", extracted_feat)