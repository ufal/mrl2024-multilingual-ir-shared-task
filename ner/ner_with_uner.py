""" Partially adapted from https://github.com/UniversalNER/uner_code/blob/master/train_uner.py"""

import logging
import os

import evaluate
import numpy as np
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer, pipeline, RobertaTokenizer

logging.basicConfig(
    format="%(asctime)s: %(message)s", level=logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
LABEL_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']


def tokenize_item(sample, tokenizer: RobertaTokenizer):
    # align label IDs with subword tokens
    tokenized = tokenizer(sample["tokens"], is_split_into_words=True, return_tensors="pt", truncation=True)

    word_ids = tokenized.word_ids(batch_index=0)
    previous_word_idx = None
    # transform NER tags to IDs
    label_ids = [LABEL_NAMES.index(tag) if "DATE" not in tag else 0 for tag in sample["ner_tags"]]
    i = -1
    aligned_labels = []

    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            aligned_labels.append(-100)

        elif word_idx != previous_word_idx:  # Normally you'd only label the first token of a given word.
            i = i + 1
            aligned_labels.append(label_ids[i])
            previous_word_idx = word_idx

        else:
            aligned_labels.append(label_ids[i])  # but in their training code, UNER did it for the other tokens as well.

    return {'input_ids': tokenized['input_ids'].squeeze(), 'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': aligned_labels}


def run_uner(data_file: str, tags_file: str, uner_model_path: str):
    classifier = pipeline("ner", model=uner_model_path)

    with open(data_file) as f_in, open(tags_file, "w") as f_out:
        for line in f_in:
            line = line.strip()
            line = " ".join(line.split())

            entities = classifier(line)
            token_starts = [0]
            for pos, char in enumerate(line):
                if char == " ":
                    token_starts.append(pos + 1)
            assert len(token_starts) == len(line.split())
            tags = ["O"] * len(line.split())

            current_tag_start = None
            current_tag_end = None
            current_entity_type = None

            def resolve_entity():
                token_start_idx = None
                token_end_idx = None
                for idx, start in enumerate(token_starts):
                    if start >= current_tag_start:
                        token_start_idx = idx
                        break
                if token_start_idx is None:
                    token_start_idx = len(token_starts)

                for idx, start in enumerate(token_starts):
                    if start >= current_tag_end:
                        token_end_idx = idx
                        break
                if token_end_idx is None:
                    token_end_idx = len(token_starts)
                for idx in range(token_start_idx, token_end_idx):
                    if idx == token_start_idx:
                        tags[idx] = "B-" + current_entity_type
                    else:
                        tags[idx] = "I-" + current_entity_type

            for entity in entities:
                tag, tag_type = entity["entity"].split("-")
                if tag == "B":
                    if current_tag_start is not None:
                        resolve_entity()
                    current_entity_type = tag_type
                    current_tag_start = entity["start"]
                    current_tag_end = entity["end"]
                elif tag == "I":
                    current_tag_end = entity["end"]
                else:
                    raise ValueError(f"Unexpected tag: {tag}")

            if current_tag_start is not None:
                resolve_entity()

            print(" ".join(tags), file=f_out)


def tune_uner(train_data: Dataset, test_data: Dataset, uner_model_path: str, model_save_path: str):
    logging.info("Loading the UNER model.")
    tokenizer = AutoTokenizer.from_pretrained(uner_model_path)
    model = AutoModelForTokenClassification.from_pretrained(uner_model_path)

    train_data = [tokenize_item(sample, tokenizer) for sample in train_data]
    test_data = [tokenize_item(sample, tokenizer) for sample in test_data]
    trainer = setup_trainer(model, tokenizer, train_data, test_data)

    trainer.train()
    base_path = os.path.basename(model_save_path)
    os.makedirs(base_path, exist_ok=True)
    trainer.save_model(model_save_path)


def compute_metrics(predictions_labels):
    metric = evaluate.load("seqeval")
    predictions, labels = predictions_labels
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [LABEL_NAMES[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_NAMES[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
        if k not in flattened_results.keys():
            flattened_results[k + "_f1"] = results[k]["f1"]

    return flattened_results


def setup_trainer(model, tokenizer, train_data, test_data):
    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_args = TrainingArguments(
        output_dir="./universal-ner",
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="linear",  # cosine
        warmup_ratio=0.1,
        dataloader_num_workers=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        # gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=500,
        run_name="uner_train",
        push_to_hub=False,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


if __name__ == "__main__":
    from collect_data import collect_data

    train_data = collect_data("all", "train")
    dev_data = collect_data("all", "dev")

    tune_uner(train_data, dev_data, "universalner/uner_all", "./output/tuned-ner")
