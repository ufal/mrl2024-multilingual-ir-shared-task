#!/bin/bash

gpus="$1"
echo "$gpus gpus available"

conda env list
conda activate snake_env

# python download.py
# python scoring.py

torchrun --nproc_per_node $gpus training.py