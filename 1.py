import pandas as pd


data = pd.read_csv(r"pl\clip_durations.tsv", sep="\t")

print(data["duration[ms]"].quantile(0.9))