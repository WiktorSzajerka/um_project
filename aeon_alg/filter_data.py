import pandas as pd

age_groups = ["teens", "twenties", "thirties", "fourties"]
age_data = pd.read_csv(r"pl\validated.tsv", sep="\t")
age_data = age_data.drop(columns=["client_id", "sentence_id", "sentence", "sentence_domain", "up_votes", "down_votes", "accents", "variant", "locale", "segment"])
age_data = age_data.dropna(subset=["age", "gender"])

# reduce dominant classes
mask = age_data["age"].isin(age_groups)
df_to_reduce = age_data[mask]
df_keep = age_data[~mask]

# Sample 1000 per age group 
df_sampled = (
    df_to_reduce
    .groupby("age")[["path","age","gender"]]
    .apply(lambda x: x.sample(n=min(1000, len(x))))
)


age_data = pd.concat([df_keep, df_sampled]).reset_index(drop=True)
age_data.loc[age_data["age"].isin(["sixties", "seventies", 'eighties', "nineties"]), "age"] = "sixties+"
age_data.to_csv("reduced_data.csv")