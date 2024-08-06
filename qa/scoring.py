import ipdb
import torch
import os
import pandas as pd
import json
import random
import sys
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm
from itertools import permutations
from __init__ import validation_folder, collection_mt_folder, results_folder, prompt_lang_mapping, get_model_path_by_name 

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", default="question_level", type=str, help="The name of the evaluation strategy, can be answer_level, question_level")
parser.add_argument("--scope", default="valid_translated", type=str, help="The name of the set to be evaluated. Can be one of valid_native, valid_translated.")
parser.add_argument("--model_name", default="llama_3.1_base", type=str, help="The name of the model to be used. Can be one of aya_101_hf, llama_3.0_base, llama_3.0_large, llama_3.1_base.")


from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

def get_llama3_yes_logit(model, tokenizer, text, device, yes_id):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs)
        # model.generate(**inputs)
    
    next_logits = output.logits[0, -1]
    yes_logit = next_logits[yes_id].item()

    final_logit = yes_logit
    # TODO: save also the entire logit as a binary file to play with it later
    return final_logit


def apply_inference_answer_level(sample, prompt_mapping, model_name, model, tokenizer, device, yes_id):
    text = sample["text"]
    question = sample["question"]
    
    for answer_id in ["A", "B", "C", "D"]:
        answer = sample[answer_id]

        header_0 = prompt_mapping.get("header_0")
        header_1 = prompt_mapping.get("header_1")
        header_2 = prompt_mapping.get("header_2")
        header_3 = prompt_mapping.get("header_3")
        
        full_text = f"{header_0} {text} {header_1} {question} {header_2} {answer}? {header_3}"
        answer_score = get_llama3_yes_logit(model, tokenizer, full_text, device, yes_id)
        sample[f"{answer_id}_score"] = answer_score
    return sample

def get_llama3_letter_logits(model, tokenizer, messages, add_header, answer_tokens, terminators):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    additional_ids = tokenizer.encode(add_header, return_tensors="pt").to(model.device)
    input_ids_plus = torch.hstack([input_ids, additional_ids[:, 1:]])

    with torch.no_grad():
        output = model(input_ids=input_ids_plus)
        next_logits = output.logits[0, -1]

    scores = {}
    for answer_id, token_id in answer_tokens.items():
        answer_score = next_logits[token_id].item()
        scores[f"{answer_id}_score"] = answer_score

    del output
    with torch.no_grad():
        outputs = model.generate(
            input_ids_plus,
            max_new_tokens=15,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    
    scores["generated_text"] = tokenizer.decode(outputs[0, input_ids_plus.shape[1]:])
    return scores
    
def get_aya_letter_logits(model, tokenizer, messages, add_header, answer_tokens):
    input_dict = tokenizer(messages[1]["content"], return_tensors="pt").to(model.device)
    additional_ids = tokenizer.encode(add_header, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model(**input_dict, decoder_input_ids=additional_ids)
        next_logits = output.logits[0, -1]

    scores = {}
    for answer_id, token_id in answer_tokens.items():
        answer_score = next_logits[token_id].item()
        scores[f"{answer_id}_score"] = answer_score

    del output
    with torch.no_grad():
        outputs = model.generate(
            **input_dict,
            max_new_tokens=15,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    scores["generated_text"] = tokenizer.decode(outputs[0]) # skip_special_tokens=True
    return scores

def apply_multiple_choice(sample, prompt_mapping, model_name, model, tokenizer, device, answer_tokens, terminators):
    text = sample.get("text", "")
    question = sample.get("question")

    sys_header = prompt_mapping.get("sys_head")
    add_header = prompt_mapping.get("add_head")

    options = []
    for answer_id in ["A", "B", "C", "D"]:
        answer = sample[answer_id]
        options.append(f"{answer_id}) {answer}")
    options_str = " ".join(options)

    messages = [
        {
            "role": "system", 
            "content": sys_header
        },
        {
            "role": "user", 
            "content": f"{text} {question} {options_str}".strip()
        }, 
    ]
    if model_name.startswith("llama"):
        answer_scores = get_llama3_letter_logits(model, tokenizer, messages, add_header, answer_tokens, terminators)  
    
    elif model_name.startswith("aya"):
        answer_scores = get_aya_letter_logits(model, tokenizer, messages, add_header, answer_tokens)  
    sample.update(answer_scores)
    return sample

def shuffle_answers(df):
    all_permutations = list(permutations(range(4)))
    chosen_permutations = [random.choice(all_permutations) for _ in range(len(df))]
    
    samples = []
    for i, row in df.iterrows():
        sample = json.loads(row.to_json())
        permutation = chosen_permutations[i]

        answers = [sample[answer_id] for answer_id in ["A", "B", "C", "D"]]
        for j in range(4):
            new_answer_id = chr(permutation[j] + ord("A"))
            sample[new_answer_id] = answers[j]
        
        label_id = ord(sample["label"]) - ord("A")
        new_label_id = permutation[label_id]
        sample["label"] = chr(new_label_id + ord("A"))

        sample['permutation'] = list(permutation)
        samples.append(sample)

    return samples

def apply_inference(model, tokenizer, device, language_ids, file_fn, strategy, terminators, csv_sep, english_prompts, using_s_tok):
    for language_id in language_ids:
        print("Starting to score language: ", language_id)

        df_path = file_fn(language_id)
        df = pd.read_csv(df_path, sep=csv_sep)
        samples = shuffle_answers(df)

        lang_code = "EN" if english_prompts else language_id
        prompt_mapping = prompt_lang_mapping.get(lang_code, {})

        yes_str = " " + prompt_mapping.get("yes").title()
        yes_id = tokenizer.vocab.get(yes_str)
        if yes_id is None:
            token_ids = tokenizer.encode(yes_str)
            yes_id = token_ids[1]

        answer_token_ids = {}
        for answer_id in ["A", "B", "C", "D"]:
            token_ids = tokenizer.encode(" " + answer_id + ")")
            token_idx = int(using_s_tok)
            answer_token_ids[answer_id] = token_ids[token_idx]

        final_samples = []
        for sample in tqdm(samples):
            if strategy == "answer_level":
                sample = apply_inference_answer_level(sample, prompt_mapping, args.model_name, model, tokenizer, device, yes_id)
            elif strategy == "question_level":
                sample = apply_multiple_choice(sample, prompt_mapping, args.model_name, model, tokenizer, device, answer_token_ids, terminators)
            
            final_samples.append(sample)

        scores_path = os.path.join(results_folder, os.path.basename(file_fn(language_id)) + "_scores.tsv")
        pd.DataFrame(final_samples).to_csv(scores_path, sep='\t', index=False)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_original_path, model_local_path = get_model_path_by_name(args.model_name)
    
    if args.model_name.startswith("llama"):
        model_class = AutoModelForCausalLM
        using_s_tok = True
    elif args.model_name.startswith("aya"):
        model_class = AutoModelForSeq2SeqLM
        using_s_tok = False

    model = model_class.from_pretrained(model_local_path, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_original_path, use_auth_token=True)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    if args.scope == "valid_native":
        language_ids = ["ALS", "AZ", "IG", "TR", "YO"]
        file_fn = lambda language_id: os.path.join(validation_folder, f"val_labeled_MC_{language_id}.csv")
        csv_sep = ','
        english_prompts = False

    elif args.scope == "valid_translated":
        language_ids = ["ALS", "AZE", "IBO", "TUR", "YOR"]
        file_fn = lambda language_id: os.path.join(collection_mt_folder, f"mrl.{language_id}.val.tsv")
        csv_sep = '\t'
        english_prompts = True

    apply_inference(model, tokenizer, device, language_ids, file_fn, args.strategy, terminators, csv_sep, english_prompts, using_s_tok)


def chat_inference(model, tokenizer, device):
    prompt_mapping = prompt_lang_mapping.get("YO")

    messages = [
        {
            "role": "system", 
            "content":  prompt_mapping.get("sys_head")
        },
        {
            "role": "user", 
            "content": "Aisha Augie-Kuta (ti a bi ni ọjọ kẹrinla oṣu kẹrin ọdun 1980) jẹ oluyaworan ati oṣere fiimu ti orilẹ-ede Naijiria ti o da ni Ilu Abuja . Arabinrin naa ni Hausa lati ijoba ibile Argungu ni ariwa Nigeria. O gba ẹbun naa fun Oluṣọọda Ẹlẹda ti ọdun ni ọdun 2011 The Future Awards .  . Augie-kuta ni Onimọnran Pataki ti isiyi (Ọgbọn Awọn ibaraẹnisọrọ oni nọmba) si Minisita fun Iṣuna-owo ati Eto Ilu. Ṣaaju si eyi o jẹ Oluranlọwọ pataki pataki fun Gomina ti Ipinle Kebbi, Nigeria lori Media Titun. Augie-Kuta ṣe itọsọna ọpọlọpọ awọn ipilẹ idagbasoke fun agbawi ti ọdọ ati ifiagbara fun awọn obinrin kaakiri Nigeria. Ekun wo ni Naijiria ni Aisha Augie-Kuta ti wa? A) Ẹkun  Ila-orun  B) Ẹkun  Guusu C) Ẹkun  Hausa D) Ẹkun Ariwa"
        }, 
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # additional_ids = tokenizer.encode(prompt_mapping.get("add_head"), return_tensors="pt").to(model.device)
    # input_ids_plus = torch.hstack([input_ids, additional_ids[:, 1:]])

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        # outputs = model(input_ids=input_ids_plus)
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=15,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        ipdb.set_trace()
        print(outputs, file=sys.stderr)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
