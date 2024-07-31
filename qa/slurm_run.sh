#!/bin/bash

job_name='AndreiLLM_job'
out_file='slurm_logs/%x.%j.out'
err_file='slurm_logs/%x.%j.err'
gpu_group='gpu-troja,gpu-ms'
gpus='1'
gram_size='gpuram48G'
mem_cpu='40G'
nodes='1'
priority='high'

sbatch -J $job_name -o $out_file -e $err_file -p $gpu_group --gpus=$gpus --constraint=$gram_size --mem=$mem_cpu -N$nodes -q $priority execute.sh $gpus