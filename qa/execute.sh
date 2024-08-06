#!/bin/bash

gpus="$1"
echo "$gpus gpus available"

conda env list
conda activate snake_env

# python download.py

SCOPE=valid_native
MODEL_NAME=llama_3.1_base

# python scoring.py --scope $SCOPE --model_name $MODEL_NAME
# python metric.py --scope $SCOPE

torchrun --nproc_per_node $gpus training.py --model_name $MODEL_NAME