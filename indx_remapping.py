import pandas as pd


age_data = pd.read_csv(r"pl\validated.tsv", sep="\t")
age_data = age_data[["path", "age"]]
age_data.loc[age_data["age"].isin(["sixties", "seventies", 'eighties', "nineties"]), "age"] = "sixties+"
age_data.dropna(inplace=True)

ind = age_data.index
mapping_df = pd.DataFrame(ind)


for i in range(10):
    age_data = pd.read_csv(rf"splits\fold{i}test.csv", index_col=0)
    remap_dict = dict(zip(mapping_df.iloc[:,0], mapping_df.index))
    age_data.index = age_data.index.map(lambda x: remap_dict.get(x, x))
    age_data.to_csv(rf"splits\fold{i}test.csv")

for i in range(10):
    age_data = pd.read_csv(rf"splits\fold{i}train.csv", index_col=0)
    remap_dict = dict(zip(mapping_df.iloc[:,0], mapping_df.index))
    age_data.index = age_data.index.map(lambda x: remap_dict.get(x, x))
    age_data.to_csv(rf"splits\fold{i}train.csv")