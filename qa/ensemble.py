# only for multiple choice questions

import os
import ipdb
import pandas as pd
import numpy as np

from __init__ import mc_qa_native_languages, results_folder

ensembe_models = [
    "aya_101_hf",
    "llama_3.0_large",
    "llama_3.1_large"
]
scores_list = [f'{answer_id}_score' for answer_id in ["A", "B", "C", "D"]]
col_rename = {score: i for i, score in enumerate(scores_list)}
columns = ["annotation_id"] + scores_list

def np_softmax(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def apply_hard_voting(dataframes):
    agg_df = pd.DataFrame()
    agg_df["annotation_id"] = dataframes[0]["annotation_id"].astype(int)

    for i, df in enumerate(dataframes):
        count_df = df[scores_list].rename(columns=col_rename)
        agg_df[i] = count_df.idxmax(axis=1)
    
    ids = list(range(len(dataframes)))
    
    hard_decisions = [np.argmax(np.bincount(line)) for line in agg_df[ids].values.tolist()]
    decision_letters = [chr(ord("A") + decision) for decision in hard_decisions]

    agg_df["prediction"] = decision_letters
    return agg_df
    

def apply_soft_voting(dataframes):
    agg_df = pd.DataFrame()
    agg_df["annotation_id"] = dataframes[0]["annotation_id"].astype(int)

    aggregator = np.zeros((len(dataframes), len(dataframes[0]), 4), dtype=np.float32)
    for i, df in enumerate(dataframes):
        aggregator[i] = np_softmax(df[scores_list].values)
    
    mean_scores = np.mean(aggregator, axis=0)
    soft_decisions = np.argmax(mean_scores, axis=1).tolist()
    decision_letters = [chr(ord("A") + decision) for decision in soft_decisions]

    agg_df["prediction"] = decision_letters
    return agg_df


def apply_voting(voting_type):
    language_ids = mc_qa_native_languages
    for lang_code in language_ids:
        dataframes = []
        
        for model_name in ensembe_models:
            df_path = os.path.join(results_folder, model_name, f"MC_{lang_code}_test.predict_scores.tsv")
            df = pd.read_csv(df_path, sep='\t')

            dataframes.append(df[columns].astype(float))

        if voting_type == "hard":
            agg_df = apply_hard_voting(dataframes)

        elif voting_type == "soft":
            agg_df = apply_soft_voting(dataframes)

        df["prediction"] = agg_df["prediction"]
        for key in scores_list + ['generated_text']:
            if key in df.keys():
                df.pop(key)

        ensemble_folder = os.path.join(results_folder, f"ensemble_{voting_type}")
        os.makedirs(ensemble_folder, exist_ok=True)

        submit_path = os.path.join(ensemble_folder, f"MC_{lang_code}_test.predict")
        df.to_csv(submit_path, sep=',', index=False)

def main():
    apply_voting("hard")
    apply_voting("soft")

if __name__ == "__main__":
    main()