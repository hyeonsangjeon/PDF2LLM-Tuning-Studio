"""Optional SFT training stage (CPU smoke by default).

Runs between ``export`` and ``eval`` when ``train.enabled`` is set. It consumes
the exported SFT rows, trains a tiny pinned model for a few steps with the shared
``pdf_qa.training`` engine, saves the model under the run's artifacts, and
records a small train report. The GPU/8B path is delegated to the quantization
track (see quantization_adapter); this stage is the credential-free proof.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from pdf_qa.run_bundle import sha256_canonical

from .stages.harness import Stage, StageContext


def _train_sig(ctx: StageContext):
    tcfg = ctx.config.get("train", {})
    return {
        "stage": "train_smoke",
        "export": sha256_canonical(ctx.outputs["export"]),
        "train": {
            "model": tcfg.get("model"),
            "max_steps": tcfg.get("max_steps"),
            "max_seq_len": tcfg.get("max_seq_len"),
            "learning_rate": tcfg.get("learning_rate"),
            "seed": (ctx.config.get("seeds") or [0])[0],
        },
    }


def _train_run(ctx: StageContext) -> Dict[str, Any]:
    from pdf_qa.training import train_sft, evaluate_sft

    tcfg = ctx.config.get("train", {})
    train_path = os.path.join(ctx.run_dir, ctx.outputs["export"]["path"])
    out_dir = os.path.join(ctx.run_dir, "artifacts", "sft_model")
    seed = (ctx.config.get("seeds") or [0])[0]

    metrics = train_sft(
        train_path=train_path,
        model_id=tcfg.get("model", "sshleifer/tiny-gpt2"),
        out_dir=out_dir,
        max_steps=int(tcfg.get("max_steps", 3)),
        max_seq_len=int(tcfg.get("max_seq_len", 128)),
        learning_rate=float(tcfg.get("learning_rate", 5e-4)),
        seed=int(seed),
    )

    with open(train_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    sample = rows[: min(3, len(rows))]
    gen = evaluate_sft(out_dir, sample, max_seq_len=int(tcfg.get("max_seq_len", 128)))

    metrics["out_dir"] = os.path.relpath(out_dir, ctx.run_dir)
    metrics["sample_generations"] = gen["generations"]
    return metrics


def make_train_stage() -> Stage:
    return Stage("train_smoke", _train_sig, _train_run)
