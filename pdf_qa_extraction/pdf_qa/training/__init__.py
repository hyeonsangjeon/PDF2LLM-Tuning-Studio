"""Reusable post-training (SFT) engine.

The CPU-smoke path here proves the export -> train -> evaluate handoff without a
GPU using a tiny pinned model and completion-only masking. The production GPU
path (LoRA / QAT at 8B scale) is delegated to the existing ``quantization`` track
via its stable CLI. This module never imports the workflow package.
"""

from .sft import format_chat, train_sft, evaluate_sft  # noqa: F401
