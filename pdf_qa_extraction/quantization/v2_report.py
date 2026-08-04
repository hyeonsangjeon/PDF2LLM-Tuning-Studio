"""Benchmark aggregation automation (spec P1-2).

The committed derived tables under ``quantization/results/`` — ``vllm_throughput.json``
(serving sweep + latency + crossover/ratios) and ``three_way_table.json`` (per-seed
EM/F1/ppl -> mean+/-std) — were originally produced by ad-hoc scripts that never made
it into the repo. That left the *derived* numbers un-reproducible and hand-editable.

This module regenerates both tables **from the read-only raw JSON inputs** with a single
command, so a human never edits a derived number by hand:

  * ``--emit``            regenerate under ``runs/<run_id>/quantization/report/`` (never
                          touches the committed ``results/`` files) + a ``provenance.json``
                          recording input file SHA-256 hashes and the exact run arguments.
  * ``--check-historical`` regenerate in memory and hash/diff against the committed
                          ``results/*.json``; exit non-zero on any mismatch, printing the
                          differing JSON pointers. The historical files are never modified.

Raw (read-only) inputs
  three_way : results/eval_<method>_seed<seed>.json      (per-seed eval rows)
  throughput: results/bench_A.json, bench_int4.json        (batch sweeps)
              results/bench_A_lat.json, bench_int4_lat.json (single-stream latency)
              results/bench_meta.json                       (engine stack + clean weight VRAM)

Derived numbers (unit-tested): int4-vs-bf16 single-stream throughput ratio, e2e/TTFT
speedups, batched ratio at max batch, the bf16<->int4 crossover batch, and the VRAM
saving factor.  Import-light (PyYAML + stdlib only); no torch.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))          # quantization/
_ROOT = os.path.dirname(_HERE)                              # pdf_qa_extraction/
DEFAULT_CONFIG = os.path.join(_HERE, "config.yaml")
DEFAULT_RESULTS = os.path.join(_HERE, "results")
DEFAULT_RUNS = os.path.join(_HERE, "runs")

# Static descriptive text (NOT measurements) — kept verbatim so the regenerated
# throughput document is byte-for-byte semantically identical to the historical one.
_TP_BENCHMARK = "vllm_v2_throughput_sweep_and_latency"
_TP_PURPOSE = (
    "Apples-to-apples serving benchmark for the 3-way comparison on REAL artifacts (W3/W4 fix). "
    "Single vLLM engine + identical knobs; batch-size sweep (1..256) to locate the int4<->bf16 "
    "crossover; single-stream TTFT + e2e p50/p99; clean serving-only weight VRAM (W5)."
)
_TP_WEIGHT_INDEP = (
    "Throughput depends only on architecture + numeric precision + serving format, NOT on trained "
    "weight VALUES. A(bf16) uses the real merged seed42 artifact; the int4 row uses the real "
    "B_int4_ptq seed42 artifact and represents BOTH B (PTQ) and C (QAT), which serialize to the "
    "identical torchao int4 tile-packed format (identical throughput by construction)."
)
_A_REPRESENTS = "Method A (BF16 LoRA) architecture + precision (real merged seed42 artifact)"
_A_MODEL = "Qwen/Qwen3-8B (LoRA-merged)"
_BC_REPRESENTS = (
    "Method B (INT4 PTQ) AND Method C (INT4 QAT) — identical torchao int4 tile-packed serving "
    "format (throughput identical by construction)"
)
_BC_MODEL = "Qwen3-8B -> torchao Int4WeightOnlyConfig(group_size=128, TILE_PACKED_TO_4D)"
_KNOBS_NOTE = (
    "ignore_eos + max_tokens=128 forces exactly 128 decode tokens/request (identical decode "
    "work). CUDA graphs on. Native sampler (FlashInfer JIT needs nvcc, absent). Latency = "
    "client-observed single-stream (batch=1): TTFT via max_tokens=1, e2e via max_tokens=128."
)
_WINNER_SINGLE = (
    "split: int4 wins throughput + full-response e2e (memory-bandwidth-bound decode at "
    "batch=1); bf16 wins TTFT (faster prefill — int4 prefill pays a dequant tax)"
)
_WINNER_BATCHED = "bf16 (compute-bound; bf16 tensor-core GEMM outruns int4 dequant+gemm)"


# --------------------------------------------------------------------------- utils
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def canonical(obj: Any) -> str:
    """Formatting-independent canonical form for hash/diff comparison."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


# ------------------------------------------------------- derived-number primitives
def mean_std(vals: List[Optional[float]]) -> Dict[str, Any]:
    """Population std (ddof=0), rounded to 3 dp — matches v2_pipeline.aggregate_seeds."""
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return {"mean": float("nan"), "std": 0.0, "n": 0}
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
    return {"mean": round(mean, 3), "std": round(std, 3), "n": len(vals)}


def aggregate_seeds(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    agg: Dict[str, Any] = {}
    for method, rs in by_method.items():
        agg[method] = {
            "base_model": rs[0]["base_model"], "precision": rs[0].get("precision", ""),
            "n_eval": rs[0]["n_eval"], "seeds": sorted(r["seed"] for r in rs),
            "exact_match": mean_std([r["exact_match"] for r in rs]),
            "f1": mean_std([r["f1"] for r in rs]),
            "perplexity": mean_std([r["perplexity"] for r in rs]),
            "size_gb": rs[0].get("size_gb"),
            "tok_per_s": mean_std([r["tok_per_s"] for r in rs]),
        }
    return agg


def _thr(sweep: List[Dict[str, Any]], batch: int) -> Optional[float]:
    for r in sweep:
        if r["batch"] == batch:
            return r["throughput_tok_s"]
    return None


def crossover_batch(sweep_bf16: List[Dict[str, Any]], sweep_int4: List[Dict[str, Any]]) -> Optional[int]:
    """Smallest batch at which bf16 throughput overtakes int4 (compute-bound regime)."""
    for r in sweep_bf16:
        b = r["batch"]
        tb, ti = _thr(sweep_bf16, b), _thr(sweep_int4, b)
        if tb is not None and ti is not None and tb > ti:
            return b
    return None


def vram_saving_x(vram_bf16: float, vram_int4: float, ndigits: int = 1) -> Optional[float]:
    return round(vram_bf16 / vram_int4, ndigits) if vram_int4 else None


# ------------------------------------------------------------------- report: throughput
def _sweep_rows(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"batch": r["batch"], "throughput_tok_s": r["throughput_tok_s"],
             "elapsed_s": r["elapsed_s"], "gen_tokens": r["gen_tokens"]} for r in d["sweep"]]


def build_throughput(results_dir: str) -> Tuple[Dict[str, Any], List[str]]:
    """Regenerate the vllm_throughput document from raw bench inputs."""
    inputs = ["bench_A.json", "bench_int4.json", "bench_A_lat.json",
              "bench_int4_lat.json", "bench_meta.json"]
    p = {name: os.path.join(results_dir, name) for name in inputs}
    sweepA, sweepI = load_json(p["bench_A.json"]), load_json(p["bench_int4.json"])
    latA, latI = load_json(p["bench_A_lat.json"]), load_json(p["bench_int4_lat.json"])
    meta = load_json(p["bench_meta.json"])
    vram_bf16 = meta["weight_vram_gib"]["A_bf16"]
    vram_int4 = meta["weight_vram_gib"]["BC_int4"]

    b1_i, b1_b = _thr(sweepI["sweep"], 1), _thr(sweepA["sweep"], 1)
    bmax = max(r["batch"] for r in sweepA["sweep"])
    bm_i, bm_b = _thr(sweepI["sweep"], bmax), _thr(sweepA["sweep"], bmax)
    xs = crossover_batch(sweepA["sweep"], sweepI["sweep"])
    ttft_b = latA["latency_single_stream"]["ttft_p50_s"]
    ttft_i = latI["latency_single_stream"]["ttft_p50_s"]
    e2e_b = latA["latency_single_stream"]["e2e_p50_s"]
    e2e_i = latI["latency_single_stream"]["e2e_p50_s"]
    vsave = vram_saving_x(vram_bf16, vram_int4)

    derived = {
        "single_stream_throughput_int4_vs_bf16": round(b1_i / b1_b, 3) if b1_b else None,
        "single_stream_e2e_int4_faster_x": round(e2e_b / e2e_i, 3) if e2e_i else None,
        "single_stream_ttft_bf16_faster_x": round(ttft_i / ttft_b, 3) if ttft_b else None,
        "single_stream_winner": _WINNER_SINGLE,
        "batched_bf16_vs_int4_at_max_batch": round(bm_b / bm_i, 3) if bm_i else None,
        "batched_winner": _WINNER_BATCHED,
        "crossover_batch_bf16_overtakes": xs,
        "takeaway": (
            f"int4 wins single-stream throughput + full-response e2e ({round(e2e_b / e2e_i, 2)}x faster: "
            f"{e2e_i}s vs {e2e_b}s) and memory ({vsave}x smaller weights: {vram_int4} vs {vram_bf16} GiB), "
            f"but bf16 has ~{round(ttft_i / ttft_b, 1)}x lower TTFT ({ttft_b}s vs {ttft_i}s — int4 prefill "
            f"pays a dequant tax). bf16 wins batched throughput (crossover ~batch {xs}; "
            f"{round(bm_b / bm_i, 1)}x at batch {bmax}). Pick int4 for memory-bound / throughput-oriented "
            f"single-stream serving, bf16 for low-TTFT interactive or max batched throughput."
        ),
    }
    doc = {
        "benchmark": _TP_BENCHMARK,
        "purpose": _TP_PURPOSE,
        "weight_independence_note": _TP_WEIGHT_INDEP,
        "engine": meta["engine"],
        "matched_knobs": {
            "dtype": "bfloat16", "max_model_len": 4096, "gpu_memory_utilization": 0.85,
            "enforce_eager": False, "cuda_graphs": True, "temperature": 0.0, "max_tokens": 128,
            "ignore_eos": True, "sampler": "native (VLLM_USE_FLASHINFER_SAMPLER=0)",
            "batches": [1, 4, 16, 32, 64, 128, 256],
            "latency_reps": latA["latency_single_stream"]["n"],
            "note": _KNOBS_NOTE,
        },
        "results": {
            "A_bf16": {
                "represents": _A_REPRESENTS, "model": _A_MODEL, "quant": "none",
                "precision": "bf16", "weight_vram_gib": vram_bf16,
                "sweep": _sweep_rows(sweepA),
                "latency_single_stream": latA["latency_single_stream"],
            },
            "BC_int4": {
                "represents": _BC_REPRESENTS, "model": _BC_MODEL, "quant": "torchao",
                "precision": "int4", "weight_vram_gib": vram_int4,
                "sweep": _sweep_rows(sweepI),
                "latency_single_stream": latI["latency_single_stream"],
            },
        },
        "derived": derived,
    }
    return doc, [p[name] for name in inputs]


# -------------------------------------------------------------- report: three-way table
def build_three_way(results_dir: str, config_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """Regenerate the per-seed EM/F1/ppl mean+/-std table from raw eval rows."""
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    base = cfg["base_model"]["selected"]
    expected_n = int(cfg["data"]["eval_size"])
    expected_seeds = set(cfg.get("seeds", []))
    rows: List[Dict[str, Any]] = []
    used: List[str] = []
    for fn in sorted(os.listdir(results_dir)):
        if fn.startswith("eval_") and fn.endswith(".json"):
            r = load_json(os.path.join(results_dir, fn))
            if (r.get("seed") in expected_seeds and r.get("n_eval") == expected_n
                    and r.get("base_model") == base):
                rows.append(r)
                used.append(os.path.join(results_dir, fn))
    doc = {
        "base_model": base, "eval_size": expected_n, "seeds": sorted(expected_seeds),
        "per_seed": rows, "aggregate": aggregate_seeds(rows),
    }
    return doc, used


# ------------------------------------------------------------------------- diff helper
def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}/{i}"))
    else:
        out[prefix or "/"] = obj
    return out


def diff_pointers(a: Any, b: Any, limit: int = 20) -> List[str]:
    """Human-readable JSON-pointer level diff between regenerated (a) and historical (b)."""
    fa, fb = _flatten(a), _flatten(b)
    diffs: List[str] = []
    for k in sorted(set(fa) | set(fb)):
        if fa.get(k, "<MISSING>") != fb.get(k, "<MISSING>"):
            diffs.append(f"{k}: regenerated={fa.get(k, '<MISSING>')!r} != historical={fb.get(k, '<MISSING>')!r}")
    return diffs[:limit]


# ------------------------------------------------------------------------------- builds
def build_all(results_dir: str, config_path: str) -> Dict[str, Tuple[Dict[str, Any], List[str]]]:
    return {
        "vllm_throughput.json": build_throughput(results_dir),
        "three_way_table.json": build_three_way(results_dir, config_path),
    }


# ---------------------------------------------------------------------------------- CLI
def _check_historical(results_dir: str, config_path: str) -> int:
    rc = 0
    for name, (doc, _inputs) in build_all(results_dir, config_path).items():
        hist_path = os.path.join(results_dir, name)
        if not os.path.exists(hist_path):
            print(f"[report] MISSING historical {name}")
            rc = 1
            continue
        historical = load_json(hist_path)
        if hash_obj(doc) == hash_obj(historical):
            print(f"[report] OK  {name} — regenerated matches committed (sha {hash_obj(doc)[:12]})")
        else:
            rc = 1
            print(f"[report] MISMATCH {name} — regenerated != committed:")
            for line in diff_pointers(doc, historical):
                print(f"    {line}")
    if rc == 0:
        print("[report] OK — all derived tables reproduce from raw JSON.")
    return rc


def _emit(results_dir: str, config_path: str, runs_dir: str, run_id: str, argv: List[str]) -> int:
    out_dir = os.path.join(runs_dir, run_id, "quantization", "report")
    os.makedirs(out_dir, exist_ok=True)
    provenance: Dict[str, Any] = {
        "tool": "quantization.v2_report",
        "run_id": run_id,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": argv,
        "results_dir": os.path.relpath(results_dir, _ROOT),
        "reports": {},
    }
    for name, (doc, inputs) in build_all(results_dir, config_path).items():
        out_path = os.path.join(out_dir, name)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        provenance["reports"][name] = {
            "output_sha256": hash_obj(doc),
            "inputs": [{"path": os.path.relpath(ip, _ROOT), "sha256": sha256_file(ip)} for ip in inputs],
        }
        print(f"[report] wrote {os.path.relpath(out_path, _ROOT)} ({len(inputs)} raw inputs)")
    prov_path = os.path.join(out_dir, "provenance.json")
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, ensure_ascii=False, indent=2)
    print(f"[report] wrote {os.path.relpath(prov_path, _ROOT)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description="Regenerate derived benchmark tables from raw JSON (spec P1-2).")
    ap.add_argument("--emit", action="store_true",
                    help="write regenerated tables under runs/<run_id>/quantization/report/")
    ap.add_argument("--check-historical", action="store_true",
                    help="verify committed results/*.json reproduce from raw JSON (read-only)")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    if not args.emit and not args.check_historical:
        args.check_historical = True  # default to the read-only gate
    if args.check_historical:
        rc = _check_historical(args.results_dir, args.config)
        if rc or not args.emit:
            return rc
    run_id = args.run_id or _dt.datetime.now(_dt.timezone.utc).strftime("report-%Y%m%d-%H%M%S")
    return _emit(args.results_dir, args.config, args.runs_dir, run_id, argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
