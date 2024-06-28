from typing import List, Union

from datasets import load_dataset, concatenate_datasets, interleave_datasets

LABELS_MASAKHANER = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"]


def load_transform_uzb_gh(split: str = "train"):
    # first clone https://github.com/azhar520/NER.git into ner folder
    if split == "train":
        file = "NER/data/train_enhanced.json"
    elif split == "dev":
        file = "NER/data/dev.json"
    elif split == "test":
        file = "NER/data/test.json"
    else:
        raise ValueError("split must be 'train', 'dev', or 'test'")
    uz_gh = load_dataset("json", data_files=file, split="train")

    def transform(entry):
        tokens = entry["sentence"]
        tags = ["0"] * len(tokens)
        entities = entry["ner"]
        for entity in entities:
            idx = entity["index"]
            if len(idx) == 1:
                tags[idx[0]] = "B-" + entity["type"]
            if len(idx) > 1:
                for i, pos in enumerate(idx):
                    if i == 0:
                        tags[pos] = "B-" + entity["type"]
                    else:
                        tags[pos] = "I-" + entity["type"]
        return {"tokens": tokens, "ner_tags": tags, "lang": "uzb"}

    uz_transformed = uz_gh.map(transform, remove_columns=["ner", "sentence"])
    return uz_transformed


def load_transform_masakhaner2(lang: str, split: str):  # particularly for yor and ibo
    if split == "dev":
        split = "validation"
    dataset = load_dataset("masakhane/masakhaner2", lang, split=split)
    dataset_transformed = dataset.map(
        lambda x: {'ner_labels': [LABELS_MASAKHANER[i] for i in x['ner_tags']], "lang": lang},
        remove_columns=["ner_tags"]
    )

    dataset_transformed = dataset_transformed.rename_column("ner_labels", "ner_tags")
    return dataset_transformed


def load_transform_polyglot(lang: str, split: str):  # particularly for tr and id
    if split == "train":
        split = "train[:10000]"
    elif split == "dev":
        split = "train[10000:11000]"
    elif split == "test":
        split = "train[11000:12000]"
    else:
        raise ValueError("split must be 'train', 'dev', or 'test'")
    dataset = load_dataset("rmyeid/polyglot_ner", lang, split=split, trust_remote_code=True)

    def transform(entry):
        tokens = entry["words"]
        tags = entry["ner"]
        new_tags = tags.copy()
        inside = False
        for i, tag in enumerate(tags):
            if tag == "O":
                inside = False
                continue
            else:  # any entity tag
                if inside:  # inside was set true at a previous tag, with no intervening "O"
                    if tags[i - 1] == tag:  # double-check that it's the same tag
                        new_tags[i] = "I-" + tag
                    else:
                        new_tags[i] = "B-" + tag  # otherwise must be a new entity
                if not inside:
                    new_tags[i] = "B-" + tag
                    inside = True
        return {"tokens": tokens, "ner_tags": new_tags}

    dataset_transformed = dataset.map(transform, remove_columns=["ner", "words"])
    return dataset_transformed


def load_transform_swissner():
    de_sw = load_dataset("ZurichNLP/swissner", split="test_de")
    dataset_transformed = de_sw.map(
        lambda x: {'ner_labels': [i if "MISC" not in i else "O" for i in x['ner_tags']], "lang": "de_SW"},
        remove_columns=["ner_tags", "url"])
    dataset_transformed = dataset_transformed.rename_column("ner_labels", "ner_tags")
    return dataset_transformed


def collect_data(datasets: Union[str, List[str]], split: str = "train"):
    if datasets == "all":
        if split == "train":
            datasets = ["de_DE", "tr_polyglot", "id_polyglot", "yor_masakhaner2", "ibo_masakhaner2", "uzb_gh"]
        elif split == "dev":
            datasets = ["de_sw", "tr_polyglot", "id_polyglot", "yor_masakhaner2", "ibo_masakhaner2", "uzb_gh"]
        elif split == "test":
            datasets = ["de_sw", "tr_polyglot", "id_polyglot", "yor_masakhaner2", "ibo_masakhaner2", "uzb_gh"]
    elif isinstance(datasets, str):
        datasets = [datasets]

    fetched = []
    for dataset in datasets:
        if dataset == "de_DE":
            fetched.append((load_transform_polyglot("de", split=split)))
        if dataset == "de_sw":
            fetched.append(load_transform_swissner())
        if dataset == "tr_polyglot":
            fetched.append(load_transform_polyglot("tr", split=split))
        if dataset == "id_polyglot":
            fetched.append(load_transform_polyglot("id", split=split))
        if dataset == "yor_masakhaner2":
            fetched.append(load_transform_masakhaner2("yor", split=split))
        if dataset == "ibo_masakhaner2":
            fetched.append(load_transform_masakhaner2("ibo", split=split))
        if dataset == "uzb_gh":
            fetched.append(load_transform_uzb_gh(split=split))

    # all_datasets = concatenate_datasets(fetched).remove_columns("id").shuffle()
    # can be updated with probabilities to sub- or up-sample a bit
    # right now I'm actually leaving out the Swiss German dataset because it's so tiny :(
    all_datasets = interleave_datasets(fetched, stopping_strategy="first_exhausted").remove_columns("id").shuffle()
    return all_datasets
