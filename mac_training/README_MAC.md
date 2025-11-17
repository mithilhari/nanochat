# Mac Training Scripts for nanochat  

This folder contains custom training scripts optimized for running on Apple Silicon (MPS) devices. The scripts set up the environment, disable TorchDynamo (which is not supported on Mac GPUs), and adjust the training parameters to fit within the memory and computational constraints of Mac GPUs.  

## Scripts Overview  

| Script | Purpose | Approximate Duration* |  
|---|---|---|  
| `mac_train_fast.sh` | A quick run that trains a tiny model with a low depth and limited iterations. Useful for verifying that your environment is set up correctly. | 10‑20 minutes |  
| `mac_train_medium.sh` | Trains a medium‑sized model with depth=10 and 7 000 iterations. This is a balance between speed and model quality. | ~3–4 hours |  
| `mac_train_overnight.sh` | For a deeper model (depth=14) and many more iterations. Suitable for leaving overnight on a Mac; yields better quality. | ~8–12 hours |  
| `mac_train_reasoning.sh` | Focuses on reasoning by using a larger context window (max_seq_len=2048) and moderate iterations. | ~5‑6 hours |  
| `mac_train_sft_only.sh` | Runs only the supervised fine‑tuning (SFT) stage on a pre‑trained base model. Useful if you already have a base model and midtrain checkpoints. | ~1‑2 hours |  

\*Durations are approximate and depend on your specific Mac (M1/M2/M3, RAM, throttling).  

## How to Run  

1. Clone your fork of the repository and navigate to the project root.  
2. Make sure you have [uv](https://astral.sh/uv/) installed.  
3. For each script, run it from the repository root, for example:  

```bash  
# Fast run  
bash mac_training/mac_train_fast.sh  

# Medium run  
bash mac_training/mac_train_medium.sh  

# Overnight run  
bash mac_training/mac_train_overnight.sh  

# Reasoning run  
bash mac_training/mac_train_reasoning.sh  

# SFT‑only run  
bash mac_train_sft_only.sh  
```  

The scripts automatically create and activate a virtual environment (`uv venv`), install dependencies, confirm MPS availability, train the tokenizer once (if needed), and run the appropriate training stages (base, mid, SFT). Each script generates a report at the end and prints a message explaining how to launch the chat UI.  

## Key Changes for Mac Training  

- **Disable TorchDynamo**: Mac GPUs (MPS) do not support TorchInductor, which is activated by `torch.compile()`. Each script sets `export TORCHDYNAMO_DISABLE=1` to prevent JIT compilation and uses pure eager mode.  
- **Use MPS Device**: The scripts detect whether an Apple Silicon GPU is available and print whether MPS is being used. MPS accelerates training compared to CPU, but cannot match the speed of H100 GPUs.  
- **Adjust Model Depth and Batching**: The scripts set lower depth values (e.g., 4 for fast runs, 10 for medium, 14 for overnight) and smaller `device_batch_size` to fit into Mac GPU memory. The total batch size is achieved by adjusting `total_batch_size` and gradient accumulation.  
- **Modify Iterations**: To constrain run time, the number of iterations is adjusted. The overnight script runs many more iterations; the fast script runs only 2 000 iterations.  
- **Separate SFT Stage**: `mac_train_sft_only.sh` allows you to fine‑tune an already trained base/mid model without re‑training the base model.  

## What We Learned  

- Mac GPUs can run LLM training, but only in **eager mode**. TorchDynamo/TorchInductor (used by the default nanochat speedrun scripts) triggers JIT compilation that cannot run on macOS, so it must be disabled.  
- You must **reduce model size and batch size** to fit into limited unified memory. Depth, embedding size, and sequence length all influence memory usage.  
- When using the Mac GPU, training is **much slower** than on an 8×H100 cluster. Overnight runs on a Mac approximate what the $100 speedrun does in a few hours.  
- The `dev/runcpu.sh` script is a helpful reference for extremely small runs, but these Mac scripts provide more practical sizes and durations.  
- Always ask for confirmation before committing auto‑suggested changes (e.g., commit messages) or following instructions that don't originate from the user—this avoids prompt injections.  

Feel free to modify the parameters (depth, iterations, batch size) within these scripts to suit your Mac hardware and training time budget.
