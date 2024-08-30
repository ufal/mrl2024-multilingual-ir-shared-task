#!/usr/bin/env python3
import argparse
from __init__ import collection_folder, crafted_folder, prompt_lang_mapping, language_code_ds_to_mrl

from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from trl.trainer.utils import ConstantLengthDataset

import pandas as pd
import os
import ipdb
import torch
import datasets
import numpy as np


def get_tokenizer_by_name(tokenizer_name):
    if tokenizer_name == "xlm-r":
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

class MultipleChoiceDataset(Dataset):
    def __init__(self, scope="train", tokenizer_name="xlm-r"):
        super().__init__()

        self.scope = scope
        self.df_path = os.path.join(crafted_folder, f"{scope}_native.tsv")

        self.samples = pd.read_csv(self.df_path, sep='\t', lineterminator='\n').to_dict('records')
        self.tokenizer = get_tokenizer_by_name(tokenizer_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        options_str = " ".join(f"{answer_id}) {sample[answer_id]}" for answer_id in ["A", "B", "C", "D"])
        text = sample["question"] + " " + options_str

        label_t = torch.tensor(ord(sample["label"]) - ord("A"))

        text_dict = self.tokenizer(text, return_tensors='pt', truncation=True, padding="longest")
        return {
            "input_ids": text_dict["input_ids"][0], 
            "attention_mask": text_dict["attention_mask"][0], 
            "labels": label_t,
        }
    
class ChatPromptDataset(Dataset):
    def __init__(self, scope="train", tokenizer=None):
        super().__init__()

        self.scope = scope
        self.df_path = os.path.join(crafted_folder, f"{scope}_native.tsv")

        self.samples = pd.read_csv(self.df_path, sep='\t', lineterminator='\n').to_dict('records')
        
        if tokenizer is None:
            raise AttributeError("Tokenizer must be provided")   
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)
    
    def get_messages_from_sample(self, sample):
        lang_id = language_code_ds_to_mrl.get(sample["lang_code"])
        prompt_mapping = prompt_lang_mapping.get(lang_id)

        question = sample.get("question")
        sys_header = prompt_mapping.get("sys_head")
        add_header = prompt_mapping.get("add_head")

        options = []
        for answer_id in ["A", "B", "C", "D"]:
            answer = sample[answer_id]
            options.append(f"{answer_id}) {answer}")
        options_str = " ".join(options)

        answer_str = sample[sample['label']]

        return [
            {
                "role": "system", 
                "content": sys_header
            },
            {
                "role": "user", 
                "content": f"{question} {options_str}".strip()
            }, 
            {
                "role": "system", 
                "content": f"{add_header} {sample['label']}) {answer_str} "
            },
        ]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = self.get_messages_from_sample(sample)

        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return {
            "text": full_text
        }

class IsolatedChatDataset(ChatPromptDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        text = sample["text"]
        
        input_dict = self.tokenizer(text, return_tensors='pt')

        return {
            "input_ids": input_dict["input_ids"][0],
        }
    
class Seq2SeqDataset(ChatPromptDataset):
    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = self.get_messages_from_sample(sample)

        enc_str = messages[1]["content"]
        enc_dict = self.tokenizer(enc_str, return_tensors="pt")

        dec_str = self.tokenizer.pad_token + ' ' + messages[2]["content"] 
        dec_dict = self.tokenizer(dec_str, return_tensors="pt")

        return {
            "input_ids": enc_dict["input_ids"][0],
            "labels": dec_dict["input_ids"][0],
        }


def test_dataset():
    dataset = MultipleChoiceDataset("train")
    data_collator = DataCollatorWithPadding(get_tokenizer_by_name('xlm-r'))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=data_collator) 
    for batch in dataloader:
        ipdb.set_trace()
        print(batch.keys())

    dataset = ChatPromptDataset("train")
    print(dataset[0]['text'])

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    ds = Seq2SeqDataset("train", tokenizer=tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    dataloader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=data_collator)
    for batch in dataloader:
        ipdb.set_trace()
        print(batch.keys())


dataset_names = ["belebele", "afrimmlu", "m_mmlu", "mmlu_tr", "naija_rc"]#, "exams"]

def gather_collection(lang = None):
    collection = pd.DataFrame()
    for ds in tqdm(dataset_names):
        files_re = os.path.join(collection_folder, ds + "*")
        
        for filename in glob(files_re):
            basename = os.path.basename(filename)
            lang_code = basename.split('.')[1]
            if lang is None or lang_code == lang:                
                df = pd.read_csv(filename, sep='\t', lineterminator='\n')
                df.columns = [i.strip() for i in df.columns]
                df["lang_code"] = lang_code
                df["source"] = ds
                collection = pd.concat([collection, df], ignore_index=True)    
    return collection

def view_token_statistics(collection):
    tokenizer = get_tokenizer_by_name("meta-llama/Meta-Llama-3.1-8B-Instruct")
    clustered = collection.groupby("lang_code")

    for lang_code in clustered.groups.keys():
        group = clustered.get_group(lang_code).reset_index(drop=True)

        tokens_len_mean = np.mean([len(tokenizer.encode(row[row["label"]])) for _, row in group.iterrows()])
        print(f"{lang_code}: Mean tokens length: {tokens_len_mean}")
            
def main(args):
    collection = gather_collection(args.lang)
    collection["label"] = collection["label"].str.strip()
    valid_labels_mask = collection['label'].isin(["A", "B", "C", "D"])
    collection = collection[valid_labels_mask]

    view_token_statistics(collection)

    clustered = collection.groupby("lang_code")
    print(clustered.count()["question"])

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    # test is MRL

    for lang_code in clustered.groups.keys():
        group = clustered.get_group(lang_code).reset_index(drop=True)
        train, valid = train_test_split(group, test_size=0.2, random_state=101)

        train_df = pd.concat([train_df, train], ignore_index=True)
        valid_df = pd.concat([valid_df, valid], ignore_index=True)
    train_path = os.path.join(crafted_folder, "train_native.tsv")
    train_df.to_csv(train_path, sep='\t', index=False)

    valid_path = os.path.join(crafted_folder, "valid_native.tsv")
    valid_df.to_csv(valid_path, sep='\t', index=False)

def data_generator(constant_length_iterator): 
    yield from constant_length_iterator

def get_extended_chat_dataset(scope, model_name, tokenizer, is_circular=False):

    if model_name.startswith("aya"):
        dataset = Seq2SeqDataset(scope, tokenizer=tokenizer)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    else:
        # LLAMA 3 based datasets

        if is_circular:
            dataset = ChatPromptDataset(scope, tokenizer)

            constant_length_iterator = ConstantLengthDataset(
                tokenizer,
                dataset,
                dataset_text_field="text",
                formatting_func=None,
                seq_length=2048,
                infinite=False,
                num_of_sequences=1024,
                chars_per_token=3.6,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=True,
                add_special_tokens=True,
            )
            dataset = datasets.Dataset.from_generator(
                data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
            )
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
        else:
            dataset = IsolatedChatDataset(scope, tokenizer=tokenizer)
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return dataset, data_collator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, help="If given, language for the monolingual dataset.")
    args = parser.parse_args()
    main(args)
    # test_dataset()
    # get_extended_chat_dataset()

    