#!/bin/bash

# Alternative launch script for 2-GPU distributed training
# Uses both GPUs for training instead of running inference server

set -e  # Exit on any error

echo "Starting distributed training with 2 GPUs..."

# Use both GPUs for distributed training
echo "Launching distributed training on GPUs 0,1..."
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num-processes 2 --config-file configs/zero3.yaml train.py

echo "Training completed!" 