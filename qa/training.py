import os
import argparse
import numpy as np
import sys
import torch
import ipdb

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, XLMRobertaForSequenceClassification, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import SFTTrainer
from peft import LoraConfig

from datasets_lab import get_tokenizer_by_name, MultipleChoiceDataset, get_extended_chat_dataset
from __init__ import get_model_path_by_name

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="xlm-r", type=str, help="The name of the model, can be XLM-R")
parser.add_argument("--num_train_epochs", default=10, type=float, help="The number of training epochs")
parser.add_argument("--batch_size", default=4, type=int, help="Size of the batch during training")
parser.add_argument("--learning_rate", default=2e-4, type=float, help="Learning rate during training")
parser.add_argument("--lr_scheduler_type", default="constant", type=str, help="The type of learning rate scheduller")
parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay used by the scheduller")
parser.add_argument("--warmup_ratio", default=0.03, type=float, help="Percentage of the training time used for warmup.")


def compute_the_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0.0
    )
    print(f"Metrics were computed: acc = {acc}, prec = {prec}, rec = {rec}, f1 = {f1}", file=sys.stderr)
    return {
        "eval_precision": prec,
        "eval_recall": rec,
        "eval_f1": f1,
    }

def get_model_by_name(model_name):
    if model_name == "xlm-r":
        model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=4)
    return model

def get_classif_model_and_ds(args):
    model = get_model_by_name(args.model_name)
    tokenizer = get_tokenizer_by_name(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = MultipleChoiceDataset("train", args.model_name)
    val_dataset = MultipleChoiceDataset("valid", args.model_name)
    return model, train_dataset, val_dataset, data_collator


def get_classif_training_arguments(args):
    output_dir = os.path.join("outputs", f'{args.model_name}')
    os.makedirs(output_dir, exist_ok=True)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        report_to=['tensorboard'],
        metric_for_best_model="eval_f1",
        optim="adamw_torch",
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4 * args.batch_size,
        gradient_accumulation_steps=args.desired_batch_size // (args.batch_size * args.num_gpus),
        seed=101,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        resume_from_checkpoint=False,
    )

def main_classifier(args):
    model, train_dataset, val_dataset, data_collator = get_classif_model_and_ds(args)
    training_args = get_classif_training_arguments(args)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_the_metrics
    )
    trainer.train()

def get_llama_training_arguments(args):
    output_dir = os.path.join("outputs", f'{args.model_name}')
    os.makedirs(output_dir, exist_ok=True)

    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        fp16=False,
        bf16=False,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        save_total_limit=1,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_steps=-1,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        learning_rate=args.learning_rate,
        log_level="info",
        logging_steps=5,
        logging_strategy="steps",
        report_to=['tensorboard'],
        seed=42,
    )


def get_llama_model_and_ds(args):
    model_original_path, model_local_path = get_model_path_by_name(args.model_name)
   
    tokenizer = get_tokenizer_by_name(model_original_path)
    # set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    train_dataset, data_collator = get_extended_chat_dataset("train", tokenizer=tokenizer)
    val_dataset, _ = get_extended_chat_dataset("valid", tokenizer=tokenizer)

    return model_local_path, train_dataset, val_dataset, tokenizer, data_collator

def main_llm(args):
    # inspired from https://github.com/AvisP/LM_Finetune/blob/main/llama-3-finetune-qlora.ipynb

    model_id, train_dataset, val_dataset, tokenizer, data_collator = get_llama_model_and_ds(args)
    training_args = get_llama_training_arguments(args)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # ).to_dict()

    model_kwargs = dict(
        torch_dtype="auto",
        use_cache=False,
        device_map="auto",
        # quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )
    trainer.train()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main_llm(args)