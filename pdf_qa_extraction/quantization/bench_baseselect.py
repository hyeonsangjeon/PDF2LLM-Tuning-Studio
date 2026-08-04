"""Base-model selection harness (v2, W1-fixed) — runnable CLI.

Restores the ``bench_baseselect.py`` script that ``notebooks/00_base_select.ipynb``
references. It re-implements, as a config-driven CLI, the zero-/few-shot KorQuAD
selection that ran on the A100 VM: a proper **chat template** (``enable_thinking``
honoured), an adequate ``max_new_tokens`` budget, correct stop handling and the
**official KorQuAD char-level EM/F1** (fixing the v1 W1 harness artifact where a
32-token raw-template zero-shot scored an incredible F1 3.5).

Design contract (spec P0-4):

* Everything (candidates, chat template, ``enable_thinking``, token budget,
  few-shot source, seed, eval slice) is read from ``config.yaml``.
* A **CPU smoke** mode (``--smoke`` / ``compute.mode: cpu``) exercises the whole
  path on a tiny ungated model + tiny slice, with no GPU.
* The default output is ``runs/<run_id>/quantization/base_select.json`` — a fresh
  run **never** touches the committed ``quantization/results/base_select.json``,
  which is a read-only historical (A100 VM) artifact.
* ``--check-historical`` only *reads* the committed results file, reports its
  SHA-256 + candidate ordering + winner, and (if a fresh run is supplied)
  diffs the two — it never mutates the file, and it does not claim the historical
  JSON was produced by this CLI invocation.

The heavy ``evaluate_candidate`` path imports torch/transformers/datasets lazily,
so the pure ranking/schema/provenance helpers (and the unit tests) stay import-light.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .data_korquad import DEFAULT_CONFIG, load_config

# Committed historical selection (produced on the A100 VM, frozen). This CLI treats
# it as read-only input and writes fresh runs elsewhere.
HISTORICAL_RESULTS = os.path.join(
    os.path.dirname(__file__), "results", "base_select.json"
)
ENTRY_REQUIRED_KEYS = ("candidate", "family", "n_eval", "zeroshot", "fewshot")
SUBMETRIC_REQUIRED_KEYS = ("exact_match", "f1", "n")


# --------------------------------------------------------------------------- #
# Pure helpers (no torch) — unit-tested
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha256(path: str) -> str:
    return sha256_file(path) if os.path.isfile(path) else ""


def has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def resolve_candidates(cfg: Dict[str, Any], token: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Turn ``config.yaml`` candidate specs into a concrete run plan.

    In CPU **smoke** mode a single tiny ungated model (``base_model.smoke``, which
    ``load_config`` already folded into ``selected``) exercises the whole path.
    Otherwise a ``gated`` candidate with no HF token present falls back to its
    ungated ``fallback`` model (recording ``gated_fallback_from``), so selection
    never blocks on credentials.
    """
    if cfg.get("compute", {}).get("mode") == "cpu":
        return [{"model_id": cfg["base_model"]["selected"], "family": "smoke",
                 "gated_fallback_from": None}]
    tok = has_hf_token() if token is None else token
    plan: List[Dict[str, Any]] = []
    for c in cfg["base_model"].get("candidates", []):
        model_id, fallback_from = c["id"], None
        if c.get("gated") and not tok and c.get("fallback"):
            model_id, fallback_from = c["fallback"], c["id"]
        plan.append({"model_id": model_id, "family": c.get("family", ""),
                     "gated_fallback_from": fallback_from})
    return plan


def rank_candidates(entries: List[Dict[str, Any]], metric: str = "zeroshot",
                    field: str = "f1") -> List[Dict[str, Any]]:
    """Deterministic ranking: primary metric F1 desc, ties broken by candidate id."""
    return sorted(entries,
                  key=lambda e: (-float(e[metric][field]), str(e.get("candidate", ""))))


def select_winner(entries: List[Dict[str, Any]], metric: str = "zeroshot") -> str:
    if not entries:
        raise ValueError("no candidate entries to select from")
    return rank_candidates(entries, metric)[0]["candidate"]


def validate_entry(entry: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if a result entry is missing required schema keys."""
    for k in ENTRY_REQUIRED_KEYS:
        if k not in entry:
            raise ValueError(f"entry missing required key: {k!r}")
    for section in ("zeroshot", "fewshot"):
        sub = entry[section]
        if not isinstance(sub, dict):
            raise ValueError(f"entry[{section!r}] must be an object")
        for k in SUBMETRIC_REQUIRED_KEYS:
            if k not in sub:
                raise ValueError(f"entry[{section!r}] missing required key: {k!r}")


def validate_results(entries: List[Dict[str, Any]]) -> None:
    if not isinstance(entries, list) or not entries:
        raise ValueError("base_select results must be a non-empty list")
    for e in entries:
        validate_entry(e)


def build_provenance(cfg: Dict[str, Any], *, mode: str, fewshot_k: int,
                     config_path: str, reproduced: bool) -> Dict[str, Any]:
    """Honest run metadata. ``reproduced=False`` marks a placeholder/plan; a real
    GPU/CPU evaluation sets ``reproduced=True``."""
    return {
        "generated_by": "quantization.bench_baseselect",
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0).isoformat(),
        "mode": mode,
        "reproduced": bool(reproduced),
        "base_selected_in_config": cfg["base_model"].get("selected"),
        "data_seed": int(cfg["data"].get("seed", 42)),
        "eval_size": cfg["data"].get("eval_size"),
        "split": "selection_dev",   # P1-1: selection never touches final_holdout
        "fewshot_k": int(fewshot_k),
        "max_new_tokens": int(cfg["eval"].get("max_new_tokens", 48)),
        "enable_thinking": bool((cfg["data"].get("chat", {}) or {}).get("enable_thinking", False)),
        "config_sha256": config_sha256(config_path),
        "note": ("Fresh run written under runs/. The committed "
                 "quantization/results/base_select.json is a separate historical "
                 "A100 artifact and is NOT modified by this CLI."),
    }


def default_run_id() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_output_path(run_id: str) -> str:
    return os.path.join("runs", run_id, "quantization", "base_select.json")


# --------------------------------------------------------------------------- #
# Heavy path (torch/transformers/datasets) — only invoked for a real run
# --------------------------------------------------------------------------- #
def evaluate_candidate(cfg: Dict[str, Any], plan: Dict[str, Any], slices: Dict[str, Any],
                       *, fewshot_k: int, n_samples: int = 3) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import eval_qa as E
    from . import v2_pipeline as V

    model_id = plan["model_id"]
    sys_p = V.system_prompt(cfg)
    et = V.enable_thinking_flag(cfg)
    smoke = cfg.get("compute", {}).get("mode") == "cpu"
    # On CPU smoke the cost is dominated by long-context prefill, so cap context +
    # new tokens hard (the smoke only needs to exercise the path, not score well).
    mnt = 16 if smoke else int(cfg["eval"].get("max_new_tokens", 48))
    bs = 8 if smoke else int(cfg["eval"].get("batch_size", 16))
    max_len = int(cfg["data"].get("max_seq_len", 1024)) if smoke else 3072
    eval_ex = slices["eval"][:8] if smoke else slices["eval"]
    fewshot = slices.get("fewshot", [])[:fewshot_k]

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    kwargs: Dict[str, Any] = {}
    if torch.cuda.is_available():
        kwargs.update(device_map="cuda", torch_dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    def run(shots: List) -> Dict[str, Any]:
        prompts = [V.build_chat_prompt(tok, sys_p, ex.context, ex.question, shots, et, True)
                   for ex in eval_ex]
        refs = [ex.answers for ex in eval_ex]
        preds: List[str] = []
        ntok, t0 = 0, time.time()
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                batch = prompts[i:i + bs]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=max_len).to(model.device)
                gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                ntok += int((new != tok.pad_token_id).sum().item())
                preds.extend(V.extract_answer(r)
                             for r in tok.batch_decode(new, skip_special_tokens=True))
        dt = time.time() - t0
        scores = E.korquad_em_f1(preds, refs)
        samples = [{"q": eval_ex[i].question, "gold": eval_ex[i].answer, "pred": preds[i]}
                   for i in range(min(n_samples, len(preds)))]
        return {"exact_match": round(scores["exact_match"], 3),
                "f1": round(scores["f1"], 6), "n": scores["n"],
                "tok_s": round(ntok / dt, 1) if dt > 0 else 0.0, "samples": samples}

    zeroshot = run([])
    fewshot_res = run(fewshot) if fewshot_k else run([])
    entry = {
        "candidate": model_id, "family": plan.get("family", ""),
        "n_eval": len(eval_ex), "zeroshot": zeroshot, "fewshot": fewshot_res,
        "peak_vram_gb": round(E.peak_vram_gb(), 2) if E.peak_vram_gb() is not None else None,
    }
    if plan.get("gated_fallback_from"):
        entry["gated_fallback_from"] = plan["gated_fallback_from"]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return entry


def run_selection(cfg: Dict[str, Any], *, fewshot_k: int) -> List[Dict[str, Any]]:
    from . import v2_pipeline as V

    plans = resolve_candidates(cfg)
    # P1-1: base-model selection uses selection_dev ONLY — never the frozen final_holdout.
    slices = V.load_slices(cfg, n_fewshot=fewshot_k, split="selection_dev")
    entries: List[Dict[str, Any]] = []
    for plan in plans:
        print(f"[base-select] evaluating {plan['model_id']} "
              f"(family={plan['family']}"
              + (f", fallback<-{plan['gated_fallback_from']}" if plan['gated_fallback_from'] else "")
              + ")")
        entries.append(evaluate_candidate(cfg, plan, slices, fewshot_k=fewshot_k))
    return entries


# --------------------------------------------------------------------------- #
# --check-historical (read-only)
# --------------------------------------------------------------------------- #
def inspect_historical(path: str = HISTORICAL_RESULTS,
                       fresh: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Read-only inspection of the committed historical selection. Verifies the
    file is untouched (SHA-256 identical before/after) and, if a fresh run is
    given, diffs winners/ordering. Never writes."""
    if not os.path.isfile(path):
        return {"exists": False, "path": path}
    sha_before = sha256_file(path)
    with open(path, "r", encoding="utf-8") as fh:
        entries = json.load(fh)
    validate_results(entries)
    ranked = rank_candidates(entries)
    report: Dict[str, Any] = {
        "exists": True, "path": path, "sha256": sha_before,
        "reproduced_by_this_cli": False,
        "status": "historical_not_reproduced",
        "winner": ranked[0]["candidate"],
        "ordering": [{"candidate": e["candidate"],
                      "zeroshot_f1": round(float(e["zeroshot"]["f1"]), 3),
                      "fewshot_f1": round(float(e["fewshot"]["f1"]), 3)} for e in ranked],
    }
    if fresh is not None:
        validate_results(fresh)
        report["fresh_winner"] = select_winner(fresh)
        report["winner_matches"] = report["fresh_winner"] == report["winner"]
    # Guarantee we never mutated the historical file.
    assert sha256_file(path) == sha_before, "historical results file changed unexpectedly"
    report["sha256_unchanged"] = True
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="quantization.bench_baseselect",
        description="Zero-/few-shot KorQuAD base-model selection (v2, config-driven).")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU smoke: tiny ungated model + tiny slice (compute.mode=cpu)")
    ap.add_argument("--fewshot", type=int, default=2, help="number of few-shot exemplars")
    ap.add_argument("--run-id", default=None, help="run id (default: UTC timestamp)")
    ap.add_argument("--out", default=None,
                    help="output path (default: runs/<run_id>/quantization/base_select.json)")
    ap.add_argument("--list-candidates", action="store_true",
                    help="print the resolved candidate plan (no model load) and exit")
    ap.add_argument("--check-historical", action="store_true",
                    help="read-only: report the committed results/base_select.json "
                         "SHA + ordering + winner; never writes")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args([] if argv is None else argv)
    force_mode = "cpu" if args.smoke else None
    cfg = load_config(args.config, force_mode=force_mode)
    mode = cfg["compute"]["mode"]

    if args.list_candidates:
        plan = resolve_candidates(cfg)
        print(json.dumps({"mode": mode, "hf_token_present": has_hf_token(),
                          "candidates": plan,
                          "selected_in_config": cfg["base_model"].get("selected")},
                         ensure_ascii=False, indent=2))
        return 0

    if args.check_historical:
        report = inspect_historical()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    run_id = args.run_id or default_run_id()
    out = args.out or default_output_path(run_id)
    if os.path.abspath(out) == os.path.abspath(HISTORICAL_RESULTS):
        raise SystemExit(
            "refusing to overwrite the historical results/base_select.json; "
            "fresh runs must write under runs/ (drop --out or pick another path)")

    entries = run_selection(cfg, fewshot_k=args.fewshot)
    validate_results(entries)
    ranked = rank_candidates(entries)
    winner = ranked[0]["candidate"]
    provenance = build_provenance(cfg, mode=mode, fewshot_k=args.fewshot,
                                  config_path=args.config, reproduced=True)

    _write_json(out, entries)
    _write_json(out.replace(".json", ".provenance.json"),
                {"provenance": provenance, "winner": winner,
                 "ordering": [e["candidate"] for e in ranked]})
    print(f"[base-select] winner = {winner}")
    print(f"[base-select] wrote {out} ({len(entries)} candidates, mode={mode})")
    cfg_sel = cfg["base_model"].get("selected")
    if mode == "gpu" and cfg_sel and winner != cfg_sel:
        print(f"[base-select] NOTE: winner {winner} != config selected {cfg_sel}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
