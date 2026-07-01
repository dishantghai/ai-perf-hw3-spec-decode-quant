#!/bin/bash
set -e
source /data/hw3/vllm_venv/bin/activate
python /data/hw3/speculators/scripts/launch_vllm.py \
    /data/hw3/Qwen3-8B-FP8-Dynamic \
    --hidden-states-path /data/hw3/hidden_states_fp8 \
    -- \
    --port 8000
