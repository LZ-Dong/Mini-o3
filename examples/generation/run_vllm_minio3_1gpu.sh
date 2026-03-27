#!/usr/bin/env bash
set -euo pipefail

# One-GPU vLLM launch script for Mini-o3 local model.
# Usage:
#   bash examples/generation/run_vllm_minio3_1gpu.sh
# Optional env:
#   MODEL_PATH=/data4/home/models/Mini-o3-7B-v1
#   PORT=8000
#   GPU_ID=0

MODEL_PATH=${MODEL_PATH:-/data4/home/models/Mini-o3-7B-v1}
PORT=${PORT:-8000}
GPU_ID=${GPU_ID:-0}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "[Mini-o3 vLLM] model=${MODEL_PATH} port=${PORT} gpu=${GPU_ID}"

# Keep naming consistent with the demo script default model path,
# and constrain image count per prompt to 10.
vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --limit-mm-per-prompt image=10 \
  --trust-remote-code
