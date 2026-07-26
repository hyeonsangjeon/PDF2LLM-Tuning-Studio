"""Shared eval scaffold for the quantization track (A/B/C use the SAME function).

Metrics:
  * KorQuAD **official** EM / F1 — Korean normalization + character-level F1
    (a naive SQuAD word-level EM would be wrong for Korean; see normalize_answer).
  * perplexity of the held-out contexts (standard quantization-paper axis).
  * model size on disk (GB), peak inference VRAM (GB), single-stream tok/s.

Output: results/<method>_metrics.json, and a helper to append a row to the 3-way table.
"""
from __future__ import annotations

import json
import os
import re
import string
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ----------------------------------------------------------------------------
# KorQuAD 1.0 official metric (ported from the KorQuAD evaluate-v1.0 script).
# ----------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """KorQuAD normalization: strip Korean quotes/brackets, punctuation, lowercase,
    collapse whitespace. (Matches the official evaluate-v1.0.py.)"""
    def remove_special(text: str) -> str:
        for ch in ["'", '"', "《", "》", "<", ">", "〈", "〉",
                    "(", ")", "‘", "’", "“", "”", "[", "]", "「", "」", "『", "』"]:
            text = text.replace(ch, " ")
        return text

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_special(s))))


def _char_tokens(text: str) -> List[str]:
    """Character-level tokens (whitespace removed) — KorQuAD F1 is char-level."""
    return [c for c in normalize_answer(text).replace(" ", "")]


def f1_score(prediction: str, ground_truth: str) -> float:
    pred = _char_tokens(prediction)
    gold = _char_tokens(ground_truth)
    if len(pred) == 0 or len(gold) == 0:
        return float(pred == gold)
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def _metric_max_over_ground_truths(metric_fn: Callable, prediction: str, ground_truths: List[str]) -> float:
    return max((metric_fn(prediction, gt) for gt in (ground_truths or [""])), default=0.0)


def korquad_em_f1(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """predictions[i] vs references[i] (list of acceptable gold answers)."""
    assert len(predictions) == len(references)
    em = f1 = 0.0
    for pred, refs in zip(predictions, references):
        em += _metric_max_over_ground_truths(exact_match_score, pred, refs)
        f1 += _metric_max_over_ground_truths(f1_score, pred, refs)
    n = max(len(predictions), 1)
    return {"exact_match": 100.0 * em / n, "f1": 100.0 * f1 / n, "n": len(predictions)}


# ----------------------------------------------------------------------------
# Generation + perplexity + resource measurement (need a loaded HF model).
# ----------------------------------------------------------------------------

def _extract_answer(generated_tail: str) -> str:
    """Take the model continuation after the prompt and keep the first answer line."""
    text = generated_tail.strip()
    for stop in ["\n[문맥]", "\n[질문]", "\n[답]", "\n\n"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip().split("\n")[0].strip()


def generate_answers(model, tokenizer, prompts: List[str], max_new_tokens: int = 32,
                     batch_size: int = 8, device: Optional[str] = None) -> Dict[str, Any]:
    """Greedy-generate answers for prompts; returns answers + single-stream tok/s."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    answers: List[str] = []
    total_new_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=2048).to(device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            total_new_tokens += int((gen != tokenizer.pad_token_id).sum().item())
            for row in tokenizer.batch_decode(gen, skip_special_tokens=True):
                answers.append(_extract_answer(row))
    elapsed = time.time() - t0
    return {"answers": answers, "tok_per_s": (total_new_tokens / elapsed) if elapsed > 0 else 0.0,
            "gen_seconds": elapsed, "new_tokens": total_new_tokens}


def perplexity(model, tokenizer, texts: List[str], max_len: int = 1024,
               device: Optional[str] = None) -> float:
    """Mean token-level perplexity over the given texts (teacher forcing)."""
    import math
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    nll_sum, tok_sum = 0.0, 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            out = model(input_ids, labels=input_ids)
            n_tok = input_ids.shape[1] - 1
            nll_sum += float(out.loss.item()) * n_tok
            tok_sum += n_tok
    if tok_sum == 0:
        return float("nan")
    return math.exp(nll_sum / tok_sum)


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


def peak_vram_gb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return None


@dataclass
class EvalResult:
    method: str
    base_model: str
    exact_match: float = 0.0
    f1: float = 0.0
    n_eval: int = 0
    perplexity: float = float("nan")
    size_gb: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    tok_per_s: float = 0.0
    precision: str = ""
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def evaluate_model(model, tokenizer, eval_examples, *, method: str, base_model: str,
                   model_dir: Optional[str] = None, max_new_tokens: int = 32,
                   batch_size: int = 8, ppl_samples: int = 200, precision: str = "",
                   notes: str = "") -> EvalResult:
    """Run the full A/B/C-shared eval on a loaded model + held-out examples.

    eval_examples: list of objects with .prompt, .answers, .context attributes
                   (quantization.data_korquad.QAExample).
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    prompts = [ex.prompt for ex in eval_examples]
    references = [ex.answers for ex in eval_examples]
    gen = generate_answers(model, tokenizer, prompts, max_new_tokens=max_new_tokens,
                           batch_size=batch_size)
    scores = korquad_em_f1(gen["answers"], references)

    ppl_texts = [ex.context for ex in eval_examples[:ppl_samples]]
    ppl = perplexity(model, tokenizer, ppl_texts)

    return EvalResult(
        method=method, base_model=base_model,
        exact_match=round(scores["exact_match"], 3), f1=round(scores["f1"], 3),
        n_eval=scores["n"], perplexity=round(ppl, 4) if ppl == ppl else float("nan"),
        size_gb=round(dir_size_gb(model_dir), 4) if model_dir and os.path.isdir(model_dir) else None,
        peak_vram_gb=(round(peak_vram_gb(), 4) if peak_vram_gb() is not None else None),
        tok_per_s=round(gen["tok_per_s"], 2), precision=precision, notes=notes,
        extra={"gen_seconds": round(gen["gen_seconds"], 2),
               "sample_predictions": gen["answers"][:3]},
    )


def write_metrics(result: EvalResult, results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"{result.method}_metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, ensure_ascii=False, indent=2)
    return out


def append_to_table(result: EvalResult, results_dir: str,
                    table_name: str = "three_way_table.json") -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, table_name)
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
    rows = [r for r in rows if r.get("method") != result.method]
    rows.append(asdict(result))
    order = {"A_bf16": 0, "B_int4_ptq": 1, "C_int4_qat": 2}
    rows.sort(key=lambda r: order.get(r.get("method", ""), 9))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    return path


def _selftest() -> None:
    """Deterministic CPU self-test of the KorQuAD metric (no model needed)."""
    preds = ["세종대왕", "1443년", "훈민정음 해례본"]
    refs = [["세종", "세종대왕"], ["1443년", "1443"], ["훈민정음"]]
    m = korquad_em_f1(preds, refs)
    print("[eval selftest] EM/F1 over 3 pairs:", m)
    assert exact_match_score("세종대왕", "세종대왕") == 1.0
    assert exact_match_score("세종대왕.", "세종대왕") == 1.0, "punct-normalized EM"
    assert f1_score("훈민정음 해례본", "훈민정음") > 0.5, "char-level partial credit"
    assert f1_score("완전히다른답", "세종대왕") == 0.0
    # max-over-refs: pred matches the 2nd gold exactly
    assert korquad_em_f1(["세종대왕"], [["세종", "세종대왕"]])["exact_match"] == 100.0
    print("[eval selftest] OK")


def load_model_for_eval(model_dir: str, precision: str = "fp32"):
    """Load a merged/base model + tokenizer for evaluation (GPU if available, else CPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(precision, torch.float32)
    tok = AutoTokenizer.from_pretrained(model_dir)
    device_map = "auto" if torch.cuda.is_available() else None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=device_map)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map=device_map)
    if not torch.cuda.is_available():
        model = model.to("cpu")
    return model, tok


def _run_cli() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model-dir", default=None, help="default = paths.method_a_dir")
    ap.add_argument("--method", default="A_bf16")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    from .data_korquad import DEFAULT_CONFIG, load_config, load_korquad

    cfg = load_config(args.config or DEFAULT_CONFIG, force_mode="cpu" if args.smoke else None)
    data = load_korquad(cfg)
    model_dir = args.model_dir or cfg["paths"]["method_a_dir"]
    model, tok = load_model_for_eval(model_dir, cfg["train"]["precision"])
    res = evaluate_model(
        model, tok, data["eval"], method=args.method,
        base_model=cfg["base_model"]["selected"], model_dir=model_dir,
        max_new_tokens=int(cfg["eval"]["max_new_tokens"]), batch_size=int(cfg["eval"]["batch_size"]),
        ppl_samples=int(cfg["eval"]["ppl_samples"]), precision=cfg["train"]["precision"],
        notes=f"mode={cfg['compute']['mode']} backend={cfg['train']['backend']}",
    )
    out = write_metrics(res, cfg["paths"]["results_dir"])
    append_to_table(res, cfg["paths"]["results_dir"])
    print("[eval] wrote", out)
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        _selftest()
    else:
        _run_cli()
