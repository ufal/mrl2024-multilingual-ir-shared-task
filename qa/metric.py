import pandas as pd
import ipdb
import os
import numpy as np
import argparse

from sklearn.metrics import accuracy_score
from nltk.translate.chrf_score import sentence_chrf
from rouge_score import rouge_scorer
from evaluate import load
from transformers import AutoTokenizer

from __init__ import results_folder, mc_qa_native_languages, mc_qa_translated_languages, open_qa_native_languages

parser = argparse.ArgumentParser()
parser.add_argument("--scope", default="valid_native", type=str, help="The name of the set to be evaluated. Can be one of valid_native, valid_translated.")
parser.add_argument("--question_type", default="multiple_choice", type=str, help="The type of the question to be evaluated. Can be one of multiple_choice, open.")
parser.add_argument("--answer_source", default="scores", type=str, help="The source of the answers. Can be one of scores, generated.")
parser.add_argument("--model_name", default="llama_3.1_base", type=str, help="The name of the model to be used. Can be one of aya_101_hf, llama_3.0_base, llama_3.0_large, llama_3.1_base.")

def compute_accuracy(language_ids, file_fn, answer_source):
    acc_list = []
    for language_id in language_ids:
        scores_path = file_fn(language_id)
        df = pd.read_csv(scores_path, sep='\t')

        predicted = []
        for _, row in df.iterrows():
            index_max = row[["A_score", "B_score", "C_score", "D_score"]].astype(float).idxmax()
            
            if answer_source == "scores":
                answer = index_max[0]
            else:
                text = row["generated_text"].replace("<pad>", "")
                answer = next(filter(str.isalpha, text))
                
            predicted.append(answer)
            
        labels = df["label"].tolist()
        acc = accuracy_score(labels, predicted)
        acc_list.append(acc)

        print(f"Accuracy for {language_id}: {acc}")
    print(f"Mean Accuracy for all languages: ", np.mean(acc_list))

def compute_string_metrics(language_ids, file_fn):
    rouger = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    bertscorer = load("bertscore")

    for language_id in language_ids:
        scores_path = file_fn(language_id)
        df = pd.read_csv(scores_path, sep='\t').fillna("")

        answers = df["answer"]
        predicted_list = []

        chrf_list = []
        rouge_l_list = []
        for _, row in df.iterrows():
            answer = row["answer"]
            predicted = row["generated_text"].replace("<|eot_id|>", "").replace("<pad>", "")
            predicted_list.append(predicted)

            chrf = sentence_chrf(answer, predicted)
            chrf_list.append(chrf)

            rouge_l = rouger.score(answer, predicted)["rougeL"].fmeasure
            rouge_l_list.append(rouge_l)
        
        bert_scores = bertscorer.compute(references=answers, predictions=predicted_list, lang=language_id.lower())['f1']
        
        chrf_mean = np.mean(chrf_list)
        rouge_l_mean = np.mean(rouge_l_list)
        bert_scores_mean = np.mean(bert_scores)

        print(f"{language_id}: chfr = {chrf_mean}, rouge_l = {rouge_l_mean}, bert_score = {bert_scores_mean}")

def check_token_stats(language_ids, file_fn):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    for language_id in language_ids:
        scores_path = file_fn(language_id)
        df = pd.read_csv(scores_path, sep='\t').fillna("")

        answers = df["answer"].tolist()
        tokens_len_mean = np.mean([len(tokenizer.encode(answer)) for answer in answers])
        print(f"{language_id}: Mean tokens length: {tokens_len_mean}")



def main(args):
    if args.question_type == "multiple_choice":

        if args.scope == "valid_native":
            language_ids = mc_qa_native_languages
            file_fn = lambda language_id: os.path.join(results_folder, args.model_name, f"val_labeled_MC_{language_id}.csv_scores.tsv")
        
        elif args.scope == "valid_translated":
            language_ids = mc_qa_translated_languages
            file_fn = lambda language_id: os.path.join(results_folder, args.model_name, f"mrl.{language_id}.val.tsv_scores.tsv")

        compute_accuracy(language_ids, file_fn, args.answer_source)

    elif args.question_type == "open":

        if args.scope == "valid_native":
            language_ids = open_qa_native_languages
            file_fn = lambda language_id: os.path.join(results_folder, args.model_name, f"QA_{language_id}_Val.csv_scores.tsv")

        check_token_stats(language_ids, file_fn)
        compute_string_metrics(language_ids, file_fn)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)