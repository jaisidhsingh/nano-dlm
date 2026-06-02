#!/bin/bash

# remember to change the permissions of this file to executable via
# `chmod +x run.sh`

source ~/.bashrc
source /lustre/fast/fast/jsingh/envs/miniconda3/etc/profile.d/conda.sh
conda activate jax
echo "Conda profile sourced and environment activated"

cd /home/jsingh/projects/nano-dlm
nvidia-smi

mkdir -p /fast/jsingh/logs/nano-dlm/err
mkdir -p /fast/jsingh/logs/nano-dlm/log
mkdir -p /fast/jsingh/logs/nano-dlm/out

python train.py
