#!/bin/bash
#SBATCH --partition="gpu-troja,gpu-ms"
#SBATCH -J llama_mono
#SBATCH -o llama_mono.out
#SBATCH -e llama_mono.err
#SBATCH -D .                      # change directory before executing the job   
#SBATCH -N 1                      # number of nodes (default 1)
#SBATCH --mem=64G                 # memory per nodes
#SBATCH --gpus=1
#SBATCH --constraint="gpuram40G|gpuram48G"
set -e

which python
date +"%Y-%m-%d %H:%M:%S"
for lang in YOR ALS AZE IBO TUR ; do
    echo "=== $LANG ==="
    # Make dataset
    echo dataset
    python datasets_lab.py --lang $LANG

    # Train
    echo Training
    python training.py --model llama_3.1_base

    # Score 
    echo Scoring
    python scoring.py --scope valid_native --model_name llama_3.1_base --question_type multiple_choice

    # Metric
    echo Metric
    python metric.py --scope valid_native --answer_source scores --question_type multiple_choice --lang $lang

    # Move to folder
    echo Move to folder
    mkdir data/results/llama_3.1_base/$lang
    mv data/results/llama_3.1_base/*_scores.tsv data/results/llama_3.1_base/$lang
done