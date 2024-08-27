#!/bin/bash

gpus="$1"
echo "$gpus gpus available"

conda env list
conda activate snake_env

gpus="$1"
echo "$gpus gpus available"

SCOPE=valid_native
MODEL_NAME=aya_101_hf
QUESTION_TYPE=multiple_choice

export OMP_NUM_THREADS=1

# torchrun --nproc_per_node $gpus --num_gpus $gpus 
python scoring.py --scope $SCOPE --model_name $MODEL_NAME --question_type $QUESTION_TYPE
python metric.py --scope $SCOPE --answer_source scores --question_type $QUESTION_TYPE

# torchrun --nproc_per_node $gpus training.py --model_name $MODEL_NAME --num_train_epochs 8 