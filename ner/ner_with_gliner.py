import logging
import os

import torch
import wandb
from datasets import Dataset
from gliner import GLiNER
from gliner.data_processing import GLiNERDataset
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.training import Trainer, TrainingArguments
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


def tune_gliner(train_data: Dataset, dev_data: Dataset, gliner_model_path: str, model_save_path: str,
                learning_rate=1e-5, epochs=5, weight_decay=0.01, test_data=None):
    logging.info("Loading the GLiNER model.")
    model = GLiNER.from_pretrained(gliner_model_path)
    model._keys_to_ignore_on_save = None

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    train_data = [ner_tags_to_spans(sample) for sample in train_data]
    dev_data = [ner_tags_to_spans(sample) for sample in dev_data]

    train_dataset = GLiNERDataset(train_data, model.config, data_processor=model.data_processor)
    dev_dataset = GLiNERDataset(dev_data, model.config, data_processor=model.data_processor)

    data_collator = DataCollatorWithPadding(model.config)

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",  # cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=2,
        label_names=["ner"],
        load_best_model_at_end=True,
        dataloader_num_workers=8,
        use_cpu=False,
        overwrite_output_dir=True,
        run_name="gliner_tuning"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    base_path = os.path.basename(model_save_path)
    os.makedirs(base_path, exist_ok=True)
    trainer.save_model(model_save_path)

    if test_data:
        test_data = [ner_tags_to_spans(sample) for sample in test_data]
        test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)
        test_loss = trainer.evaluate(test_dataset)['eval_loss']
        return test_loss
    else:
        return None


def run_gliner(f_input, f_output, gliner_model_path):
    logging.info("Loading the GLiNER model.")
    model = GLiNER.from_pretrained(gliner_model_path)
    labels = ["person", "organization", "location"]
    logging.info("Model loaded, processing the input file.")
    for l_no, line in enumerate(f_input):
        line = line.strip()
        line = " ".join(line.split())

        # Get token starts and ends
        token_starts = [0]
        for pos, char in enumerate(line):
            if char == " ":
                if line[pos + 1] == " ":
                    continue  # should avoid an issue where some sentences have random double spaces
                token_starts.append(pos + 1)
        assert len(token_starts) == len(line.split())
        tags = ["O"] * len(line.split())

        # Predict entities
        entities = model.predict_entities(line, labels)
        # print(f"Line: {line}")
        for entity in entities:
            char_start = entity["start"]
            char_end = entity["end"]

            # Find token indices
            token_start_idx = None
            for idx, start in enumerate(token_starts):
                # hacky fix for untokenized years in brackets, which led to a couple exceptions:
                if start == char_start - 1 and line[start] == "(":
                    token_start_idx = idx
                    break
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

            assert token_start_idx is not None, f"token_start_idx is None! Line: {line} Entity: {entity}"
            assert token_start_idx <= token_end_idx, f'start_idx {token_start_idx} > end_idx {token_end_idx}. Line: {line} Entity: {entity}'

            if entity["label"] == "person":
                tag = "PER"
            elif entity["label"] == "organization":
                tag = "ORG"
            elif entity["label"] == "location":
                tag = "LOC"
            else:
                raise ValueError(f"Unknown entity label: {entity['label']}")
            if token_start_idx >= len(tags):
                print(
                    f"ERROR: token_start_idx {token_start_idx} is >= len(tags) {len(tags)}. Line: {line} Entity: {entity}")
            tags[token_start_idx] = "B-" + tag
            for idx in range(token_start_idx + 1, token_end_idx):
                if idx >= len(tags):
                    print(
                        f"WARNING: Trying to modify an index larger than len(tags)! Will skip. idx: {idx}. Line: {line}. Entity: {entity}")
                    continue
                tags[idx] = "I-" + tag

        if l_no % 10 == 9:
            logging.info(f"Processed {l_no + 1} lines.")
        print(" ".join(tags), file=f_output)
    logging.info("GliNER processing finished.")


def main_sweep():
    wandb.init(project="gliner-sweep")
    loss = sweep_train(wandb.config)
    wandb.log({"loss": loss})


def sweep_train(config):
    train_data = collect_data(config.datasets, "train")
    dev_data = collect_data(config.datasets, "dev")
    loss = tune_gliner(train_data, dev_data, gliner_model_path="urchade/gliner_multi-v2.1",
                       model_save_path="./gliner-multi-tuned-sweep", learning_rate=config.learning_rate,
                       epochs=config.epochs, weight_decay=config.weight_decay, test_data=test_data)
    return loss


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "my-first-sweep"

    test_data = collect_data(["de_sw", "tr_polyglot", "yor_masakhaner2", "ibo_masakhaner2", "aze"], "test")
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "learning_rate": {
                "distribution": "uniform",
                "min": 0.00001,
                "max": 0.00004
            },
            "epochs": {"min": 3, "max": 5},
            "weight_decay": {
                "distribution": "uniform",
                "min": 0,
                "max": 0.02
            },
            "datasets": {
                "values": [
                    ["de_DE", "tr_polyglot", "yor_masakhaner2", "ibo_masakhaner2", "aze"],
                    ["de_DE", "de_SW", "tr_polyglot", "yor_masakhaner2", "ibo_masakhaner2", "aze"],
                    ["de_DE", "tr_polyglot", "id_polyglot", "yor_masakhaner2", "ibo_masakhaner2", "uzb_gh", "aze"]
                ]
            },
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
    wandb.agent(sweep_id, function=main_sweep, count=10)

