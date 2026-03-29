#!/usr/bin/env bash
set -euo pipefail

# One-GPU vLLM launch script for SenseNova-MARS local model.
# Usage:
#   bash examples/generation/run_vllm_sensenovamars_1gpu.sh
# Optional env:
#   MODEL_PATH=/data4/home/models/SenseNova-MARS-8B
#   PORT=8000
#   GPU_ID=0

MODEL_PATH=${MODEL_PATH:-/data4/home/models/SenseNova-MARS-8B}
PORT=${PORT:-8000}
GPU_ID=${GPU_ID:-1}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "[SenseNova-MARS vLLM] model=${MODEL_PATH} port=${PORT} gpu=${GPU_ID}"
  
# Keep naming consistent with the demo script default model path,
# and constrain image count per prompt to 10.
vllm serve "${MODEL_PATH}" \
  --served-model-name SenseNova-MARS-8B \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image": 12}' \
  --trust-remote-code
