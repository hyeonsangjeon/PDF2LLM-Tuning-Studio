"""v2 pipeline — chat-template, completion-only, multi-seed 3-way quantization run.

This module layers the v2 overhaul on top of the existing quantization/ code. It fixes
the five v1 weaknesses at their root:

  * W1  chat-template prompts (not the raw text template) for BOTH training and eval, so
        the strong instruct base is measured/used correctly (v1's zero-shot artifact).
  * W2  completion-only loss (loss ONLY on the answer tokens) + full-epoch A + long QAT,
        so fine-tuning actually teaches the task (v1 spent <5% of loss on the answer).
  * W5  multi-seed training -> mean +/- std on a large (2k) fixed held-out slice.

The INT4 recipe (TorchAO tile-packed int4 weight-only, group 128) and the KorQuAD official
EM/F1 metric are reused unchanged from ``eval_qa`` so B and C share an identical serving
format and the numbers stay comparable to the metric definition.

Everything is driven by ``config.yaml``. Per-seed artifacts live under
``artifacts/<Method>_seed<seed>/``.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data_korquad import DEFAULT_CONFIG, QAExample, load_config
from . import eval_qa as E

# --------------------------------------------------------------------------- #
# Prompting (chat template) + answer extraction
# --------------------------------------------------------------------------- #
FALLBACK_SYSTEM = ("당신은 한국어 독해 질의응답 어시스턴트입니다. 주어진 문맥에서 질문의 정답을 찾아 "
                   "정답에 해당하는 어구만 간결하게 출력하세요. 부연 설명이나 문장 없이 정답만 쓰세요.")


def system_prompt(cfg: Dict[str, Any]) -> str:
    return (cfg.get("data", {}).get("chat", {}) or {}).get("system", FALLBACK_SYSTEM).strip()


def enable_thinking_flag(cfg: Dict[str, Any]) -> bool:
    return bool((cfg.get("data", {}).get("chat", {}) or {}).get("enable_thinking", False))


def build_chat_prompt(tokenizer, system: str, context: str, question: str,
                      fewshot: Optional[List] = None, enable_thinking: bool = False,
                      add_generation_prompt: bool = True) -> str:
    """Render a chat prompt string via the tokenizer's chat template.

    ``fewshot`` is a list of (context, question, answer) exemplars (train split; used only
    for the base-select few-shot column — the fine-tuned models are eval'd zero-shot).
    Qwen3 accepts ``enable_thinking``; other templates raise TypeError -> retry without it.
    """
    msgs = [{"role": "system", "content": system}]
    for (c, q, a) in (fewshot or []):
        msgs.append({"role": "user", "content": f"[문맥]\n{c}\n\n[질문]\n{q}"})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"[문맥]\n{context}\n\n[질문]\n{question}"})
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


def extract_answer(text: str) -> str:
    """Keep the assistant's answer only: drop any <think> block, strip labels, first line."""
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    for lead in ("답:", "정답:", "답변:", "[답]", "답)", "A:"):
        if t.startswith(lead):
            t = t[len(lead):].strip()
    return t.split("\n")[0].strip()


# --------------------------------------------------------------------------- #
# Data slices (multi-seed): eval held-out FIXED (data.seed); train shuffled by train_seed
# --------------------------------------------------------------------------- #
def load_slices(cfg: Dict[str, Any], train_seed: Optional[int] = None,
                n_fewshot: int = 0) -> Dict[str, Any]:
    from datasets import load_dataset

    dcfg = cfg["data"]
    eval_seed = int(dcfg.get("seed", 42))
    tseed = int(train_seed if train_seed is not None else eval_seed)
    ds = load_dataset(dcfg["dataset"])

    val = ds["validation"].shuffle(seed=eval_seed)
    eval_size = dcfg.get("eval_size")
    if eval_size:
        val = val.select(range(min(int(eval_size), len(val))))

    train = ds["train"].shuffle(seed=tseed)
    subset = dcfg.get("train_subset")
    if subset:
        train = train.select(range(min(int(subset), len(train))))

    def to_ex(r):
        answers = [a for a in (r["answers"]["text"] if r.get("answers") else []) if a and a.strip()] or [""]
        return QAExample(id=str(r.get("id", "")), prompt="", answer=answers[0],
                         answers=answers, context=r["context"], question=r["question"])

    train_ex = [to_ex(r) for r in train]
    eval_ex = [to_ex(r) for r in val]
    fewshot = [(train[i]["context"], train[i]["question"], train[i]["answers"]["text"][0])
               for i in range(n_fewshot)] if n_fewshot else []
    return {"train": train_ex, "eval": eval_ex, "fewshot": fewshot,
            "train_seed": tseed, "eval_seed": eval_seed}


# --------------------------------------------------------------------------- #
# Completion-only tokenization (loss ONLY on the answer tokens) + collator
# --------------------------------------------------------------------------- #
def tokenize_completion_only(tokenizer, examples: List[QAExample], system: str,
                             max_len: int, enable_thinking: bool = False):
    from datasets import Dataset

    eos = tokenizer.eos_token_id
    input_ids_all, labels_all = [], []
    for ex in examples:
        prompt = build_chat_prompt(tokenizer, system, ex.context, ex.question,
                                   fewshot=None, enable_thinking=enable_thinking,
                                   add_generation_prompt=True)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(ex.answer, add_special_tokens=False)["input_ids"]
        if eos is not None:
            a_ids = a_ids + [eos]
        input_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + list(a_ids)
        if len(input_ids) > max_len:            # rare: left-truncate (keep the answer intact)
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]
        input_ids_all.append(input_ids)
        labels_all.append(labels)
    return Dataset.from_dict({"input_ids": input_ids_all, "labels": labels_all})


@dataclass
class DataCollatorCompletionOnly:
    pad_token_id: int
    label_pad: int = -100

    def __call__(self, feats):
        import torch

        maxlen = max(len(f["input_ids"]) for f in feats)
        ids, labs, attn = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            pad = maxlen - n
            ids.append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            labs.append(list(f["labels"]) + [self.label_pad] * pad)
            attn.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn),
                "labels": torch.tensor(labs)}


# --------------------------------------------------------------------------- #
# Method A — BF16 LoRA (chat + completion-only, full epochs, per seed)
# --------------------------------------------------------------------------- #
def train_method_a(cfg: Dict[str, Any], seed: int, out_merged: str,
                   resume: bool = False) -> Dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    base = cfg["base_model"]["selected"]
    tcfg, lcfg, dcfg = cfg["train"], cfg["lora"], cfg["data"]
    sys_p = system_prompt(cfg)
    max_len = int(dcfg["max_seq_len"])

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.enable_input_require_grads()
    peft_cfg = LoraConfig(r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]),
                          lora_dropout=float(lcfg["dropout"]),
                          target_modules=list(lcfg["target_modules"]),
                          bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)

    data = load_slices(cfg, train_seed=seed)
    train_ds = tokenize_completion_only(tok, data["train"], sys_p, max_len,
                                        enable_thinking_flag(cfg))
    run_dir = out_merged + "_run"
    ta_kwargs = dict(
        output_dir=run_dir, per_device_train_batch_size=int(tcfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(tcfg["grad_accum"]), learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=float(tcfg["warmup_ratio"]), weight_decay=float(tcfg["weight_decay"]),
        logging_steps=int(tcfg["logging_steps"]), num_train_epochs=float(tcfg["epochs"]),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)), seed=int(seed),
        bf16=True, gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        save_steps=int(tcfg.get("save_steps", 500)),
        save_total_limit=2, report_to="none", logging_first_step=True)
    if tcfg.get("max_steps"):
        ta_kwargs["max_steps"] = int(tcfg["max_steps"])
    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollatorCompletionOnly(tok.pad_token_id))
    t0 = time.time()
    ckpt = _last_checkpoint(run_dir) if resume else None
    out = trainer.train(resume_from_checkpoint=ckpt)
    secs = time.time() - t0

    os.makedirs(out_merged, exist_ok=True)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out_merged)
    tok.save_pretrained(out_merged)

    loss_curve = [(l.get("step"), l.get("loss")) for l in trainer.state.log_history if "loss" in l]
    log = {"method": "A_bf16", "base_model": base, "seed": int(seed),
           "n_train": len(data["train"]), "epochs": float(tcfg["epochs"]),
           "global_step": int(trainer.state.global_step), "train_seconds": round(secs, 1),
           "train_loss": float(getattr(out, "training_loss", float("nan"))),
           "loss_curve": loss_curve, "merged_dir": out_merged}
    del model, trainer, merged
    torch.cuda.empty_cache()
    return log


def _last_checkpoint(run_dir: str) -> Optional[str]:
    try:
        from transformers.trainer_utils import get_last_checkpoint

        return get_last_checkpoint(run_dir) if os.path.isdir(run_dir) else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Method B — INT4 PTQ (TorchAO tile-packed int4) of a merged A seed
# --------------------------------------------------------------------------- #
def build_method_b(cfg: Dict[str, Any], a_dir: str, b_dir: str) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    gs = int(cfg["ptq"]["group_size"])
    tok = AutoTokenizer.from_pretrained(a_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    qmodel = AutoModelForCausalLM.from_pretrained(
        a_dir, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=TorchAoConfig(quant_type=E.make_int4_weightonly_config(gs)))
    os.makedirs(b_dir, exist_ok=True)
    qmodel.save_pretrained(b_dir, safe_serialization=False)
    tok.save_pretrained(b_dir)
    size_gb = E.dir_size_gb(b_dir)
    del qmodel
    torch.cuda.empty_cache()
    return {"method": "B_int4_ptq", "src_A": a_dir, "int4_dir": b_dir,
            "group_size": gs, "size_gb": round(size_gb, 4)}


# --------------------------------------------------------------------------- #
# Method C — INT4 QAT (matched fake-quant STE, full-param, 8-bit Adam) of a merged A seed
# --------------------------------------------------------------------------- #
def train_method_c(cfg: Dict[str, Any], a_dir: str, c_dir: str, seed: int,
                   resume: bool = False) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, TorchAoConfig)
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs

    qcfg, dcfg = cfg["qat"], cfg["data"]
    gs = int(qcfg["group_size"])
    sys_p = system_prompt(cfg)
    max_len = int(dcfg["max_seq_len"])
    c_bf16 = c_dir + "_bf16adapt"

    def _not_lmhead(m, fqn):
        return isinstance(m, nn.Linear) and "lm_head" not in fqn

    tok = AutoTokenizer.from_pretrained(a_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a_dir, dtype=torch.bfloat16, device_map="cuda")

    _, wcfg = _infer_fake_quantize_configs(Int4WeightOnlyConfig(group_size=gs))
    quantize_(model, QATConfig(weight_config=wcfg, step="prepare"), filter_fn=_not_lmhead)

    for p in model.parameters():
        p.requires_grad_(False)
    n_train = 0
    for m in model.modules():
        if type(m).__name__ == "FakeQuantizedLinear":
            m.weight.requires_grad_(True)
            n_train += m.weight.numel()
    if qcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    data = load_slices(cfg, train_seed=seed)
    train_ds = tokenize_completion_only(tok, data["train"], sys_p, max_len,
                                        enable_thinking_flag(cfg))
    args = TrainingArguments(
        output_dir=c_dir + "_run", per_device_train_batch_size=int(qcfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(qcfg["grad_accum"]), learning_rate=float(qcfg["learning_rate"]),
        max_steps=int(qcfg["max_steps"]), warmup_ratio=0.03, logging_steps=20, seed=int(seed),
        bf16=True, optim=qcfg.get("optim", "adamw_8bit"), gradient_checkpointing=False,
        save_steps=int(qcfg.get("save_steps", 150)), save_total_limit=2,
        report_to="none", logging_first_step=True)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollatorCompletionOnly(tok.pad_token_id))
    t0 = time.time()
    ckpt = _last_checkpoint(c_dir + "_run") if resume else None
    out = trainer.train(resume_from_checkpoint=ckpt)
    secs = time.time() - t0
    model = trainer.model

    quantize_(model, QATConfig(step="convert"), filter_fn=_not_lmhead)
    os.makedirs(c_bf16, exist_ok=True)
    model.save_pretrained(c_bf16)
    tok.save_pretrained(c_bf16)
    loss_curve = [(l.get("step"), l.get("loss")) for l in trainer.state.log_history if "loss" in l]
    del model, trainer
    torch.cuda.empty_cache()

    qmodel = AutoModelForCausalLM.from_pretrained(
        c_bf16, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=TorchAoConfig(quant_type=E.make_int4_weightonly_config(gs)))
    os.makedirs(c_dir, exist_ok=True)
    qmodel.save_pretrained(c_dir, safe_serialization=False)
    tok.save_pretrained(c_dir)
    size_gb = E.dir_size_gb(c_dir)
    del qmodel
    torch.cuda.empty_cache()
    return {"method": "C_int4_qat", "src_A": a_dir, "int4_dir": c_dir, "seed": int(seed),
            "group_size": gs, "max_steps": int(qcfg["max_steps"]), "trainable_millions": round(n_train / 1e6, 1),
            "train_seconds": round(secs, 1), "train_loss": float(getattr(out, "training_loss", float("nan"))),
            "loss_curve": loss_curve, "size_gb": round(size_gb, 4)}


# --------------------------------------------------------------------------- #
# Eval (chat-aware) — reuses the official KorQuAD EM/F1 + perplexity from eval_qa
# --------------------------------------------------------------------------- #
def eval_model_chat(cfg: Dict[str, Any], model, tok, eval_ex: List[QAExample], *,
                    method: str, seed: int, model_dir: Optional[str] = None,
                    precision: str = "") -> Dict[str, Any]:
    import torch

    sys_p = system_prompt(cfg)
    et = enable_thinking_flag(cfg)
    ecfg = cfg["eval"]
    mnt, bs = int(ecfg["max_new_tokens"]), int(ecfg["batch_size"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"   # keep the question + assistant marker (prompt tail) on overflow
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompts = [build_chat_prompt(tok, sys_p, ex.context, ex.question, None, et, True) for ex in eval_ex]
    refs = [ex.answers for ex in eval_ex]
    preds, ntok, t0 = [], 0, time.time()
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=3072).to(model.device)
            gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            ntok += int((new != tok.pad_token_id).sum().item())
            preds.extend(extract_answer(r) for r in tok.batch_decode(new, skip_special_tokens=True))
    dt = time.time() - t0
    scores = E.korquad_em_f1(preds, refs)
    ppl = E.perplexity(model, tok, [ex.context for ex in eval_ex[:int(ecfg["ppl_samples"])]])
    return {"method": method, "seed": int(seed), "base_model": cfg["base_model"]["selected"],
            "exact_match": round(scores["exact_match"], 3), "f1": round(scores["f1"], 3),
            "n_eval": scores["n"], "perplexity": round(ppl, 4) if ppl == ppl else float("nan"),
            "size_gb": round(E.dir_size_gb(model_dir), 4) if model_dir and os.path.isdir(model_dir) else None,
            "peak_vram_gb": round(E.peak_vram_gb(), 4) if E.peak_vram_gb() is not None else None,
            "tok_per_s": round(ntok / dt, 2) if dt > 0 else 0.0, "precision": precision,
            "gen_seconds": round(dt, 1), "sample_predictions": preds[:3]}


def qat_scheme_selftest(device: str = "cuda", d: int = 512, gs: int = 128) -> Dict[str, Any]:
    """Guard the exact QAT prepare/convert path that ``train_method_c`` runs.

    Rubber-duck concern #1 asked whether C trains against what it serves. torchao 0.17 only
    exposes an int4 *fake-quant* for PLAIN/PRESHUFFLED packing (``_infer_fake_quantize_configs``
    rejects TILE_PACKED_TO_4D), while the only kernel-free int4 *serving* path on this box is
    TILE_PACKED_TO_4D. The two use the same asymmetric group-``gs`` int4 family but different
    kernels, so their per-layer outputs are *not* bit-identical — an exact-match assertion would
    be wrong. What actually matters, and what this checks, is:

    1. ``prepare`` truly replaces child Linears with ``FakeQuantizedLinear`` (the transform is a
       no-op on a bare root module, so this must be checked on a real *container*), i.e. QAT
       actually simulates int4 during training rather than silently training full-precision.
    2. The fake-quant rounding error is the same order of magnitude as the tile-packed serving
       error (same int4 family) — so the QAT signal transfers to the served format.
    3. ``convert`` + tile-packed re-quant round-trips without error (the export path in C).

    Returns a diagnostics dict with an ``ok`` flag (True when all three hold).
    """
    import torch
    import torch.nn as nn
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs

    def _lin(m, fqn):
        return isinstance(m, nn.Linear)

    torch.manual_seed(0)
    x = torch.randn(8, d, dtype=torch.bfloat16, device=device)

    def _mk():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(d, d, bias=False), nn.ReLU(),
                             nn.Linear(d, d, bias=False)).to(device, torch.bfloat16)

    ref = _mk()
    with torch.no_grad():
        y_orig = ref(x)

    # Real tile-packed int4 serving (what B and C actually deploy).
    serve = _mk()
    quantize_(serve, E.make_int4_weightonly_config(gs))
    with torch.no_grad():
        y_serve = serve(x)
    serve_err = float((y_orig - y_serve).norm() / (y_orig.norm() + 1e-9))

    # Exact QAT prepare path from train_method_c (fake-quant on child Linears).
    fake = _mk()
    _, wcfg = _infer_fake_quantize_configs(Int4WeightOnlyConfig(group_size=gs))
    quantize_(fake, QATConfig(weight_config=wcfg, step="prepare"), filter_fn=_lin)
    n_fq = sum(1 for m in fake.modules() if type(m).__name__ == "FakeQuantizedLinear")
    with torch.no_grad():
        y_fake = fake(x)
    fake_err = float((y_orig - y_fake).norm() / (y_orig.norm() + 1e-9))

    # Convert (strip fake-quant -> int4-robust bf16) then tile-packed re-quant: C's export.
    convert_ok = True
    try:
        quantize_(fake, QATConfig(step="convert"), filter_fn=_lin)
        quantize_(fake, E.make_int4_weightonly_config(gs))
        with torch.no_grad():
            _ = fake(x)
    except Exception:
        convert_ok = False

    ratio = fake_err / serve_err if serve_err > 0 else float("inf")
    fires = n_fq > 0
    same_family = 0.4 <= ratio <= 2.5
    ok = bool(fires and same_family and convert_ok and fake_err > 0.01)
    return {"ok": ok, "prepare_fires": fires, "fake_quant_layers": n_fq,
            "fake_err_vs_orig": round(fake_err, 5), "serve_err_vs_orig": round(serve_err, 5),
            "fake_to_serve_ratio": round(ratio, 3), "same_int4_family": same_family,
            "convert_roundtrip_ok": convert_ok, "group_size": gs}


def aggregate_seeds(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """rows: per-(method,seed) eval dicts -> {method: {mean/std for em/f1/ppl, seeds:[...]}}"""
    def ms(vals):
        vals = [v for v in vals if v is not None and v == v]
        if not vals:
            return {"mean": float("nan"), "std": 0.0, "n": 0}
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 3), "std": round(std, 3), "n": len(vals)}

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    agg = {}
    for method, rs in by_method.items():
        agg[method] = {
            "base_model": rs[0]["base_model"], "precision": rs[0].get("precision", ""),
            "n_eval": rs[0]["n_eval"], "seeds": sorted(r["seed"] for r in rs),
            "exact_match": ms([r["exact_match"] for r in rs]),
            "f1": ms([r["f1"] for r in rs]),
            "perplexity": ms([r["perplexity"] for r in rs]),
            "size_gb": rs[0].get("size_gb"),
            "tok_per_s": ms([r["tok_per_s"] for r in rs]),
        }
    return agg
