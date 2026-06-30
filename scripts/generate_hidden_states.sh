#!/bin/bash
set -e
source /data/hw3/speculators_venv/bin/activate
python /data/hw3/speculators/scripts/generate_hidden_states.py \
    --model /data/hw3/Qwen3-8B \
    --dataset sharegpt \
    --max-samples 3000 \
    --max-seq-len 2048 \
    --output-dir /data/hw3/hidden_states/ \
    --vllm-url http://localhost:8000
