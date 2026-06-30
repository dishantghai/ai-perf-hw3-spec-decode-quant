#!/bin/bash
set -e
source /data/hw3/speculators_venv/bin/activate
python /data/hw3/speculators/scripts/train_eagle3.py \
    --model /data/hw3/Qwen3-8B \
    --hidden-states-dir /data/hw3/hidden_states/ \
    --output-dir /data/hw3/output/checkpoints/ \
    --num-epochs 5 \
    --num-speculative-tokens 3
