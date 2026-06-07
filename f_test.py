import pandas as pd
import numpy as np
from scipy import stats


def F_test(a, b, cv=(5, 2)):
    n_repeats, k_folds = cv

    delta = a.reshape(n_repeats, k_folds) - b.reshape(n_repeats, k_folds)
    fold_mean = np.mean(delta, axis=-1)
    var = np.sum(np.power(delta - fold_mean[:, None], 2), axis=-1)

    f_stat = np.sum(np.power(delta, 2)) / (2 * np.sum(var))
    pvalue = stats.f.sf(f_stat, k_folds, k_folds * n_repeats)

    return f_stat, pvalue

n_cent = pd.read_csv(r"scores\nearest_centroid_scores")
rand_forest = pd.read_csv(r"scores\random_forest_scores")
svc = pd.read_csv(r"scores\svc_scores")

cls_scores = [n_cent["nearest_centroid"], rand_forest["random_forest"], svc["svc"]]
names = ["nearest_centroid", "random_forest", "svc"]


dt_results = pd.DataFrame(columns=["model1", "model2", "p-value"])
ind = 0
for i, scores in enumerate(cls_scores):
    for j in range(i, 3):
        ind += 1
        stat, p_value = F_test(np.array(scores), np.array(cls_scores[j]))
        row = [names[i], names[j], p_value]
        dt_results.loc[ind] = row
        

dt_results.to_csv(r"scores/f_test_results.csv")

