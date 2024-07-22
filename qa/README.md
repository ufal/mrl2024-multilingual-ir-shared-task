# MRL Data

## Naming convention

`{dataset}.{LANG}.{split}.tsv`

E.g., "mrl.ALS.val.tsv", "mmlu.EN.auxiliary_train.tsv"

We use "." instead of "_" or "-" as separator because some datasets use them in their names.

The language codes are the following:

- ALS: Alsatian 
- AZE: Azerbaijani
- DEU: German
- ENG: English
- IBO: Igbo
- IND: Indonesian
- TUR: Turkish
- UZB: Uzbek / Northen Uzbek
- YOR: Yoruba

## Data Format

The datasets are TSV files. 
The header is: "question" (string), "A" (string), "B" (string), "C" (string), "D" (string), "label" (A, B, C or D).
Note that some answers are the string "None" and the libraries like Pandas might interpret it as NaN.

