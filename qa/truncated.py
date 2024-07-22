""" This counts how many examples are longer the NLLB context size """
import transformers
import glob
import pandas as pd
from tqdm import tqdm

LENGTH = 1024

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    files = glob.glob("data/*.tsv")

    total = 0
    long = 0
    for file in files:
        df = pd.read_csv(file, delimiter="\t", na_values=()).fillna("")
        tokens = tokenizer(df["question"].tolist())["input_ids"]
        trunc = sum(map(lambda x: len(x) > LENGTH, tokens))
        total += len(tokens)
        long += trunc
        print(f"{file:}\t {trunc}/{len(tokens)}\t{trunc/len(tokens) * 100}%")
    print(f"--Total:\t{long}/{total}\t{long/total*100}%")
