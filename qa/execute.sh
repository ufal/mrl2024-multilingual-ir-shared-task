#!/bin/bash

gpus="$1"
echo "$gpus gpus available"

conda env list
conda activate snake_env

# python download.py

SCOPE=valid_native
MODEL_NAME=llama_3.0_large
QUESTION_TYPE=multiple_choice

python scoring.py --scope $SCOPE --model_name $MODEL_NAME --question_type $QUESTION_TYPE
python metric.py --scope $SCOPE --answer_source scores --question_type $QUESTION_TYPE

# torchrun --nproc_per_node $gpus training.py --model_name $MODEL_NAME --num_train_epochs 8 