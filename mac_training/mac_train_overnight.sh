#!/bin/bash
# mac_train_overnight.sh — overnight training for nanochat on Mac MPS

export TORCHDYNAMO_DISABLE=1

# Setup environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    uv venv
    source .venv/bin/activate
    uv sync --extra cpu
fi

# Confirm MPS availability
python - <<EOF
import torch
print("MPS available:", torch.backends.mps.is_available())
print("Using device:", "mps" if torch.backends.mps.is_available() else "cpu")
EOF

# Train tokenizer and evaluate
python -m nanochat.dataset -n 4
python -m scripts.tok_train --max_chars=200000000
python -m scripts.tok_eval

# Base model training (overnight)
python -m scripts.base_train \
  --depth=14 \
  --max_seq_len=1024 \
  --device_batch_size=2 \
  --total_batch_size=2048 \
  --eval_every=500 \
  --eval_tokens=4096 \
  --core_metric_every=1000 \
  --core_metric_max_per_task=16 \
  --sample_every=1000 \
  --num_iterations=20000

# Evaluate base model
python -m scripts.base_loss --device_batch_size=2 --split_tokens=4096
python -m scripts.base_eval --max-per-task=16

# Midtraining stage
python -m scripts.mid_train \
  --max_seq_len=1024 \
  --device_batch_size=2 \
  --eval_every=500 \
  --eval_tokens=4096 \
  --total_batch_size=2048 \
  --num_iterations=5000

python -m scripts.chat_eval --source=mid --max-new-tokens=240 --max-problems=20

# SFT fine-tuning
python -m scripts.chat_sft \
  --device_batch_size=2 \
  --target_examples_per_step=4 \
  --num_iterations=3000 \
  --eval_steps=4 \
  --eval_metrics_max_problems=16

# Generate final report
python -m nanochat.report generate

echo "Overnight training complete"
