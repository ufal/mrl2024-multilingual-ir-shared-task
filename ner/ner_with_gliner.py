import logging
import os

import torch
import wandb
from datasets import Dataset
from gliner import GLiNER, GLiNERConfig
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset

from collect_data import collect_data

logging.basicConfig(
    format="%(asctime)s: %(message)s", level=logging.INFO)


def ner_tags_to_spans(sample):
    """
    Converts NER tags in the dataset samples to spans (start, end, entity type).
    Taken from a GliNER example -> convert data format again to match the model's expectations.
    Modified because this originally assumed that the tags would be given as IDs.

    Args:
        sample (dict): A dictionary containing the tokens and NER tags.

    Returns:
        dict: A dictionary containing tokenized text and corresponding NER spans.
    """

    tags_to_gliner_names = {"B-PER": "person", "I-PER": "person", "B-ORG": "organization", "I-ORG": "organization",
                            "B-LOC": "location", "I-LOC": "location", "B-DATE": "date", "I-DATE": "date"}
    labels = ["person", "organization", "location", "date"]
    ner_tags = sample["ner_tags"]
    spans = []
    start_pos = None
    entity_name = None

    for i, tag in enumerate(ner_tags):
        if tag == "O":
            if entity_name is not None:
                spans.append((start_pos, i - 1, entity_name))
                entity_name = None
                start_pos = None
        else:
            tag_name = tag
            if tag_name.startswith('B-'):
                if entity_name is not None:
                    spans.append((start_pos, i - 1, entity_name))
                entity_name = tags_to_gliner_names[tag_name]
                start_pos = i
            elif tag_name.startswith('I-'):
                continue

    # Handle the last entity if the sentence ends with an entity
    if entity_name is not None:
        spans.append((start_pos, len(sample["tokens"]) - 1, entity_name))

    return {"tokenized_text": sample["tokens"], "ner": spans, "label": labels}


def tune_gliner(train_data: Dataset, test_data: Dataset, gliner_model_path: str, model_save_path: str):
    logging.info("Loading the GLiNER model.")
    model = GLiNER.from_pretrained(gliner_model_path)
    labels = ["person", "organization", "location", "date"]

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="mrl2024-ner",
        # set the name of the run
        name="gliner tuning",
    )

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    # model.compile_for_training()  # Dynamo is not supported on Python 3.12+

    # got a weird error, trying something else
    # train_data = train_data.map(ner_tags_to_spans)
    # test_data = test_data.map(ner_tags_to_spans)
    train_data = [ner_tags_to_spans(sample) for sample in train_data][1500:]
    test_data = [ner_tags_to_spans(sample) for sample in test_data]

    train_dataset = GLiNERDataset(train_data, model.config, data_processor=model.data_processor)
    test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)

    data_collator = DataCollatorWithPadding(model.config)

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",  # cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # label_names=["ner"],
        # load_best_model_at_end=True,
        # metric_for_best_model="f1",
        dataloader_num_workers=8,
        use_cpu=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    base_path = os.path.basename(model_save_path)
    os.makedirs(base_path, exist_ok=True)
    trainer.save_model(model_save_path)  # you know, based on what _load_best_model looks like, i'm not SURE this works


def run_gliner(f_input, f_output, gliner_model_path):
    logging.info("Loading the GLiNER model.")
    model = GLiNER.from_pretrained(gliner_model_path)
    labels = ["person", "organization", "location", "date"]
    logging.info("Model loaded, processing the input file.")
    for l_no, line in enumerate(f_input):
        line = line.strip()
        line = " ".join(line.split())

        # Get token starts and ends
        token_starts = [0]
        for pos, char in enumerate(line):
            if char == " ":
                token_starts.append(pos + 1)
        assert len(token_starts) == len(line.split())
        tags = ["O"] * len(line.split())

        # Predict entities
        entities = model.predict_entities(line, labels)

        for entity in entities:
            char_start = entity["start"]
            char_end = entity["end"]

            # Find token indices
            token_start_idx = None
            for idx, start in enumerate(token_starts):
                if start >= char_start:
                    token_start_idx = idx
                    break
            token_end_idx = None
            for idx, start in enumerate(token_starts):
                if start >= char_end:
                    token_end_idx = idx
                    break
            if token_end_idx is None:
                token_end_idx = len(token_starts)

            assert token_start_idx is not None
            assert token_start_idx <= token_end_idx

            if entity["label"] == "person":
                tag = "PER"
            elif entity["label"] == "organization":
                tag = "ORG"
            elif entity["label"] == "location":
                tag = "LOC"
            elif entity["label"] == "date":
                tag = "DATE"
            else:
                raise ValueError(f"Unknown entity label: {entity['label']}")

            tags[token_start_idx] = "B-" + tag
            for idx in range(token_start_idx + 1, token_end_idx):
                tags[idx] = "I-" + tag

        if l_no % 10 == 9:
            logging.info(f"Processed {l_no + 1} lines.")
        print(" ".join(tags), file=f_output)
    logging.info("GliNER processing finished.")
