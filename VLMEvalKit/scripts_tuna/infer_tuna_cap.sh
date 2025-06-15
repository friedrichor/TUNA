#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
num_gpu=${#GPU_IDS[@]}

export PYTHONPATH=`pwd`:$PYTHONPATH

ts=`date +%Y_%m_%d_%H_%M`
log_dir=./logs/tuna_cap/
mkdir -p $log_dir
model=Qwen2-VL-7B-Instruct


torchrun --nproc-per-node=$num_gpu --master_port=25781 run.py \
    --data TUNA_CAP_2fps \
    --model $model \
    --mode infer \
    --verbose 2>&1 | tee ${log_dir}/log_${ts}_${model}.log
