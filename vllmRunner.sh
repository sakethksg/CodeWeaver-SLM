#!/bin/bash

# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES="0,1"

# Start the vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/deepseek-coder-6.7b-instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 32768 \
  --dtype auto
