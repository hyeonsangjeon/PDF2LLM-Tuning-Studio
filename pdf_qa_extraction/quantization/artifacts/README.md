# artifacts/ — Method A outputs (NOT committed; regenerable)

Model weights are large and reproducible, so they are git-ignored. Running
`train_lora.py` (or notebook `01_bf16_lora.ipynb`) recreates them here:

| Path | What | Role |
|---|---|---|
| `A_bf16/` | **Merged BF16 model** (base + LoRA folded in) | **Input to Part 2** (INT4 PTQ/QAT) and the eval target for Method A |
| `A_bf16_adapter/` | LoRA adapter only (r16/α32) | Lightweight, re-mergeable |
| `A_bf16_run/` | trl trainer checkpoints/logs | Transient |

## Reproduce
```bash
cd pdf_qa_extraction
# GPU VM (spec): config.yaml compute.mode=gpu  -> Qwen/Qwen3-1.7B, unsloth, BF16, full data
python -m quantization.train_lora
# CPU smoke (no GPU): tiny model + subset + 12 steps
python -m quantization.train_lora --smoke
```

Part 2 (B/C) starts from `A_bf16/` — keep the path stable.
