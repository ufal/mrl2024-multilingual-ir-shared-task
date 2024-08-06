import pandas as pd
import ipdb
import os
import numpy as np
import argparse

from sklearn.metrics import accuracy_score
from __init__ import results_folder

parser = argparse.ArgumentParser()
parser.add_argument("--scope", default="valid_native", type=str, help="The name of the set to be evaluated. Can be one of valid_native, valid_translated.")


def compute_accuracy(language_ids, file_fn):
    acc_list = []
    for language_id in language_ids:
        scores_path = file_fn(language_id)
        df = pd.read_csv(scores_path, sep='\t')

        predicted = []
        for _, row in df.iterrows():
            index_max = row[["A_score", "B_score", "C_score", "D_score"]].astype(float).idxmax()
            answer = index_max[0]
            
            # answer = next(filter(str.isalpha, row["generated_text"].replace("<pad>", "")))
            predicted.append(answer)
            
        labels = df["label"].tolist()
        acc = accuracy_score(labels, predicted)
        acc_list.append(acc)

        print(f"Accuracy for {language_id}: {acc}")
    print(f"Mean Accuracy for all languages: ", np.mean(acc_list))

def main(args):
    if args.scope == "valid_native":
        language_ids = ["ALS", "AZ", "IG", "TR", "YO"]
        file_fn = lambda language_id: os.path.join(results_folder, f"val_labeled_MC_{language_id}.csv_scores.tsv")
    
    elif args.scope == "valid_translated":
        language_ids = ["ALS", "AZE", "IBO", "TUR", "YOR"]
        file_fn = lambda language_id: os.path.join(results_folder, f"mrl.{language_id}.val.tsv_scores.tsv")

    compute_accuracy(language_ids, file_fn)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)