#!/bin/bash
set -e
source /data/hw3/speculators_venv/bin/activate
python /data/hw3/speculators/scripts/train.py \
    --verifier-name-or-path /data/hw3/Qwen3-8B-FP8-Dynamic \
    --speculator-type eagle3 \
    --data-path /data/hw3/output \
    --hidden-states-path /data/hw3/hidden_states_fp8 \
    --save-path /data/hw3/output/checkpoints_fp8 \
    --epochs 5 \
    --logger tensorboard \
    --log-dir /data/hw3/logs_fp8 \
    --save-best \
    2>&1 | tee /data/hw3/output/train_run_fp8.log
