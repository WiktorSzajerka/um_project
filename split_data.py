import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

data = pd.read_csv(r"pl\validated.tsv", sep="\t")
data = data[["path", "age"]]
data.loc[data["age"].isin(["sixties", "seventies", 'eighties', "nineties"]), "age"] = "sixties+"
data.dropna(inplace=True)
targets = data["age"]
placeholder = np.zeros((targets.shape[0], 2), dtype=np.uint16)

skf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

for i, (train, test) in enumerate(skf.split(placeholder, targets)):
    data.iloc[train].to_csv(f"fold{i}train.csv")
    data.iloc[test].to_csv(f"fold{i}test.csv")