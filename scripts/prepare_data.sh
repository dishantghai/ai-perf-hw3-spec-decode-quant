#!/bin/bash
set -e
source /data/hw3/speculators_venv/bin/activate
python /data/hw3/speculators/scripts/prepare_data.py \
    --model /data/hw3/Qwen3-8B \
    --data sharegpt \
    --max-samples 3000 \
    --seq-length 2048 \
    --output /data/hw3/output
