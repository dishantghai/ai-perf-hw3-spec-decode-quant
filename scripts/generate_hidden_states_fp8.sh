#!/bin/bash
set -e
source /data/hw3/speculators_venv/bin/activate
python /data/hw3/speculators/scripts/data_generation_offline.py \
    --model /data/hw3/Qwen3-8B-FP8-Dynamic \
    --preprocessed-data /data/hw3/output \
    --output /data/hw3/hidden_states_fp8 \
    --max-samples 3000
