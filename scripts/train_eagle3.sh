#!/bin/bash
set -e
source /data/hw3/speculators_venv/bin/activate
pip install -q tensorboard
python /data/hw3/speculators/scripts/train.py \
    --verifier-name-or-path /data/hw3/Qwen3-8B \
    --speculator-type eagle3 \
    --data-path /data/hw3/output \
    --hidden-states-path /data/hw3/hidden_states \
    --save-path /data/hw3/output/checkpoints \
    --epochs 5 \
    --logger tensorboard \
    --log-dir /data/hw3/logs \
    --save-best \
    2>&1 | tee /data/hw3/output/train_run.log
