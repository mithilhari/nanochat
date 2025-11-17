#!/bin/bash
# mac_train_fast.sh — Quick Mac training script for nanochat
# Disables TorchDynamo and trains a small model quickly (~20 minutes)

export TORCHDYNAMO_DISABLE=1

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    uv venv
    source .venv/bin/activate
    uv sync --extra cpu
fi

# Confirm MPS availability
python - <<'EOF'
import torch
print("MPS available:", torch.backends.mps.is_available())
EOF

# Train tokenizer small (reduced max chars)
python -m nanochat.dataset -n 2
python -m scripts.tok_train --max_chars=50000000
python -m scripts.tok_eval

# Train base model quick (depth=4, few iterations)
python -m scripts.base_train \
  --depth=4 \
  --max_seq_len=1024 \
  --device_batch_size=2 \
  --total_batch_size=512 \
  --eval_every=100 \
  --eval_tokens=2048 \
  --core_metric_every=200 \
  --core_metric_max_per_task=8 \
  --sample_every=200 \
  --num_iterations=2000

# Evaluate base model
python -m scripts.base_loss --device_batch_size=2 --split_tokens=2048
python -m scripts.base_eval --max-per-task=8

# Midtraining quick
python -m scripts.mid_train \
  --max_seq_len=1024 \
  --device_batch_size=2 \
  --eval_every=200 \
  --eval_tokens=2048 \
  --total_batch_size=512 \
  --num_iterations=500
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=10

# SFT quick
python -m scripts.chat_sft \
  --device_batch_size=2 \
  --target_examples_per_step=4 \
  --num_iterations=500 \
  --eval_steps=4 \
  --eval_metrics_max_problems=8

# Generate report
python -m nanochat.report generate

echo "Fast training complete. Run: python -m scripts.chat_web"
