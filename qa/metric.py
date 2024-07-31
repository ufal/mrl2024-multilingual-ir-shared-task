import pandas as pd
import ipdb
import os
import numpy as np

from sklearn.metrics import accuracy_score
from __init__ import results_folder


def compute_accuracy(language_ids, file_fn):
    acc_list = []
    for language_id in language_ids:
        scores_path = file_fn(language_id)
        df = pd.read_csv(scores_path, sep='\t')

        predicted = []
        for _, row in df.iterrows():
            index_max = row[["A_score", "B_score", "C_score", "D_score"]].astype(float).idxmax()
            answer = index_max[0]
            
            # answer = next(filter(str.isalpha, row["generated_text"]))
            predicted.append(answer)
            
        labels = df["label"].tolist()
        acc = accuracy_score(labels, predicted)
        acc_list.append(acc)

        print(f"Accuracy for {language_id}: {acc}")
    print(f"Mean Accuracy for all languages: ", np.mean(acc_list))


language_ids = ["ALS", "AZ", "IG", "TR", "YO"]
file_fn = lambda language_id: os.path.join(results_folder, f"val_labeled_MC_{language_id}_scores.tsv")

# language_ids = ["ALZ", "AZE", "IBO", "TUR", "YOR"]
# file_fn = lambda language_id: os.path.join(results_folder, f"mrl.{language_id}.val.tsv_scores.tsv")

compute_accuracy(language_ids, file_fn)