import logging

from gliner import GLiNER

logging.basicConfig(
    format="%(asctime)s: %(message)s", level=logging.INFO)


def run_gliner(f_input, f_output):
    logging.info("Loading the GLiNER model.")
    model = GLiNER.from_pretrained("urchade/gliner_largev2")
    labels = ["person", "organization", "location", "date"]
    for line in f_input:
        line = line.strip()

        # Get token starts and ends
        token_starts = [0]
        for pos, char in enumerate(line):
            if char == " ":
                token_starts.append(pos + 1)
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
        print(" ".join(tags), file=f_output)
