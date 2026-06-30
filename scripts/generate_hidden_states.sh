#!/bin/bash
set -e
# speculators_venv: data_generation_offline.py imports from speculators
source /data/hw3/speculators_venv/bin/activate
python /data/hw3/speculators/scripts/data_generation_offline.py \
    --model /data/hw3/Qwen3-8B \
    --preprocessed-data /data/hw3/output \
    --output /data/hw3/hidden_states \
    --max-samples 3000
