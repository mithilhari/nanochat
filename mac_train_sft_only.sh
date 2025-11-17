#!/bin/bash
# mac_train_sft_only.sh — SFT-only training on Mac (no base or mid training)

# Disable TorchDynamo (TorchInductor not supported on macOS)
export TORCHDYNAMO_DISABLE=1

# Activate or create Python virtual environment using uv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    uv venv
    source .venv/bin/activate
    # Install dependencies with CPU extras (MPS uses CPU fallback where needed)
    uv sync --extra cpu
fi

# Confirm MPS (Apple Silicon GPU) availability
python - <<'EOF'
import torch
print("MPS available:", torch.backends.mps.is_available())
print("Using device:", "mps" if torch.backends.mps.is_available() else "cpu")
EOF

# Train tokenizer on a small subset (required once)
python -m nanochat.dataset -n 4
python -m scripts.tok_train --max_chars=200000000
python -m scripts.tok_eval

# Evaluate base model (optional check before SFT)
python -m scripts.base_loss --device_batch_size=2 --split_tokens=4096
python -m scripts.base_eval --max-per-task=16

# Run supervised fine-tuning (SFT) only. Skip base and mid training.
python -m scripts.chat_sft \
  --device_batch_size=2 \
  --target_examples_per_step=4 \
  --num_iterations=2000 \
  --eval_steps=4 \
  --eval_metrics_max_problems=16

# Evaluate the chat model after SFT
python -m scripts.chat_eval --source=sft --max-new-tokens=240 --max-problems=20

# Generate report summarizing metrics and results
python -m nanochat.report generate

echo "SFT-only training completed on Mac. You can start the chat UI with: python -m scripts.chat_web"
