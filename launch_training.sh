#!/bin/bash

# Launch script for 2-GPU setup
# Adapts the original 4-GPU commands for 2 GPUs

set -e  # Exit on any error

echo "Starting training pipeline with 2 GPUs..."

# Option 1: Run inference server on GPU 0, training on GPU 1
echo "Starting inference server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 vf-vllm --model 'Qwen/Qwen2.5-1.5B-Instruct' --tensor-parallel-size 1 &
INFERENCE_PID=$!

# Wait a bit for inference server to start
echo "Waiting for inference server to initialize..."
sleep 10

echo "Starting training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml train.py &
TRAINING_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo "Cleaning up processes..."
    kill $INFERENCE_PID $TRAINING_PID 2>/dev/null || true
    wait
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Wait for training to complete
wait $TRAINING_PID

echo "Training completed!" 