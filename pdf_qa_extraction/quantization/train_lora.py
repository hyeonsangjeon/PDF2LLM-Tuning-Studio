"""Method A — BF16 LoRA fine-tune on KorQuAD (baseline of the 3-way comparison).

Two backends behind one config knob:
  * ``train.backend: unsloth`` — GPU-only (spec default). Unsloth FastLanguageModel,
    BF16 (load_in_4bit=false), LoRA r16/a32, trl SFTTrainer.
  * ``train.backend: hf``      — transformers + peft + trl. CPU-capable, used for the
    smoke run (tiny model + subset) so every block executes when no GPU is available.

Output: LoRA adapter + a merged BF16 model at paths.method_a_dir (input to Part 2 PTQ/QAT).

CLI:
    python -m quantization.train_lora            # uses compute.mode in config.yaml
    python -m quantization.train_lora --smoke    # force CPU smoke overrides
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Tuple

from .data_korquad import DEFAULT_CONFIG, load_config, load_korquad, to_hf_text_dataset


def _dtype(precision: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        precision, torch.float32)


def build_model_and_tokenizer(cfg: Dict[str, Any]):
    """Return (model_with_lora, tokenizer, backend)."""
    backend = cfg["train"]["backend"]
    base = cfg["base_model"]["selected"]
    lcfg = cfg["lora"]
    precision = cfg["train"]["precision"]
    max_seq = int(cfg["data"]["max_seq_len"])

    if backend == "unsloth":
        from unsloth import FastLanguageModel  # GPU-only import

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base, max_seq_length=max_seq,
            dtype=_dtype(precision), load_in_4bit=bool(cfg["train"]["load_in_4bit"]),
        )
        model = FastLanguageModel.get_peft_model(
            model, r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]),
            lora_dropout=float(lcfg["dropout"]), target_modules=list(lcfg["target_modules"]),
            bias="none", use_gradient_checkpointing="unsloth", random_state=cfg["data"]["seed"],
        )
        return model, tokenizer, backend

    # hf backend (transformers + peft)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = dict(device_map=("auto" if torch.cuda.is_available() else None))
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=_dtype(precision), **load_kwargs)
    except TypeError:  # older transformers uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=_dtype(precision), **load_kwargs)
    peft_cfg = LoraConfig(
        r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]), lora_dropout=float(lcfg["dropout"]),
        target_modules=list(lcfg["target_modules"]), bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    return model, tokenizer, backend


def _build_sft_trainer(cfg, model, tokenizer, train_ds):
    """Construct a trl SFTTrainer robustly across trl / transformers versions.

    Only keys actually accepted by the installed SFTConfig are passed (the
    max_seq_length -> max_length rename and the CPU ``use_cpu`` flag differ by
    version), so this works on both the GPU (unsloth/bf16) and CPU (smoke) paths.
    """
    import dataclasses

    import torch
    from trl import SFTConfig, SFTTrainer

    tcfg = cfg["train"]
    out_dir = os.path.join(cfg["paths"]["artifacts_dir"], "A_bf16_run")
    on_gpu = torch.cuda.is_available()

    candidates = dict(
        output_dir=out_dir,
        per_device_train_batch_size=int(tcfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(tcfg["grad_accum"]),
        learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        weight_decay=float(tcfg["weight_decay"]),
        logging_steps=int(tcfg["logging_steps"]),
        num_train_epochs=float(tcfg["epochs"]),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        seed=int(cfg["data"]["seed"]),
        report_to="none",
        packing=False,
        # trl renamed max_seq_length -> max_length; provide both, filter below.
        max_length=int(cfg["data"]["max_seq_len"]),
        max_seq_length=int(cfg["data"]["max_seq_len"]),
        dataset_text_field="text",
        bf16=bool(on_gpu and cfg["train"]["precision"] == "bf16"),
        fp16=bool(on_gpu and cfg["train"]["precision"] == "fp16"),
        use_cpu=(not on_gpu),
    )
    if tcfg.get("max_steps"):
        candidates["max_steps"] = int(tcfg["max_steps"])

    valid = {f.name for f in dataclasses.fields(SFTConfig)}
    kwargs = {k: v for k, v in candidates.items() if k in valid}
    args = SFTConfig(**kwargs)

    try:
        return SFTTrainer(model=model, args=args, train_dataset=train_ds,
                          processing_class=tokenizer)
    except TypeError:
        return SFTTrainer(model=model, args=args, train_dataset=train_ds,
                          tokenizer=tokenizer)


def _save_merged(cfg, model, tokenizer, backend) -> str:
    merged_dir = cfg["paths"]["method_a_dir"]
    os.makedirs(merged_dir, exist_ok=True)
    if backend == "unsloth":
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    else:
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
    return merged_dir


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = load_korquad(cfg)
    model, tokenizer, backend = build_model_and_tokenizer(cfg)
    eos = tokenizer.eos_token or "</s>"
    train_ds = to_hf_text_dataset(data["train"], eos)

    trainer = _build_sft_trainer(cfg, model, tokenizer, train_ds)
    t0 = time.time()
    train_out = trainer.train()
    train_seconds = time.time() - t0

    adapter_dir = os.path.join(cfg["paths"]["artifacts_dir"], "A_bf16_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    merged_dir = None
    if cfg["train"].get("save_merged", True):
        merged_dir = _save_merged(cfg, trainer.model, tokenizer, backend)

    log = {
        "base_model": cfg["base_model"]["selected"],
        "backend": backend,
        "precision": cfg["train"]["precision"],
        "mode": cfg["compute"]["mode"],
        "n_train": len(data["train"]),
        "train_seconds": round(train_seconds, 2),
        "train_loss": float(getattr(train_out, "training_loss", float("nan"))),
        "global_step": int(getattr(train_out, "global_step", 0)) if hasattr(train_out, "global_step")
                       else int(getattr(getattr(trainer, "state", None), "global_step", 0)),
        "adapter_dir": adapter_dir,
        "merged_dir": merged_dir,
    }
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["results_dir"], "A_train_log.json"), "w",
              encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=2)
    return log


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, force_mode="cpu" if args.smoke else None)
    print(f"[train] base={cfg['base_model']['selected']} backend={cfg['train']['backend']} "
          f"precision={cfg['train']['precision']} mode={cfg['compute']['mode']}")
    log = train(cfg)
    print("[train] done:", json.dumps(log, ensure_ascii=False))


if __name__ == "__main__":
    _main()
