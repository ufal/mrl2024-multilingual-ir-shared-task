#!/bin/bash

gpus="$1"
echo "$gpus gpus available"

conda env list
conda activate snake_env

# python download.py

SCOPE=test_native
MODEL_NAME=llama_3.0_base
QUESTION_TYPE=open

python scoring.py --scope $SCOPE --model_name $MODEL_NAME --question_type $QUESTION_TYPE
# python metric.py --scope $SCOPE --answer_source scores

# torchrun --nproc_per_node $gpus training.py --num_gpus $gpus --model_name $MODEL_NAME --num_train_epochs 10 