"""v2 CLI orchestrator — run the multi-seed 3-way pipeline on the GPU VM.

Each subcommand runs in its OWN process (fresh CUDA + import state), so INT4/QAT
loads never collide with a training import. Per-seed artifacts:
  artifacts/A_bf16_seed<S>/  B_int4_ptq_seed<S>/  C_int4_qat_seed<S>/
Per-(method,seed) eval:      results/eval_<method>_seed<S>.json
Aggregated table:            results/three_way_table.json  (mean +/- std over seeds)

Usage:
  python -m quantization.v2_run a    --seed 42 [--resume]
  python -m quantization.v2_run b    --seed 42
  python -m quantization.v2_run c    --seed 42
  python -m quantization.v2_run eval --method A_bf16 --seed 42
  python -m quantization.v2_run agg
Smoke overrides (fast end-to-end code check): --subset --max-steps --eval-size --epochs
"""
from __future__ import annotations

import argparse
import json
import os

from .data_korquad import DEFAULT_CONFIG, load_config
from . import v2_pipeline as V


def artifact_dir(cfg, method: str, seed: int) -> str:
    base = {"A_bf16": cfg["paths"]["method_a_dir"],
            "B_int4_ptq": cfg["paths"]["method_b_dir"],
            "C_int4_qat": cfg["paths"]["method_c_dir"]}[method]
    return f"{base}_seed{seed}"


def _apply_overrides(cfg, args):
    if args.subset is not None:
        cfg["data"]["train_subset"] = args.subset
    if args.eval_size is not None:
        cfg["data"]["eval_size"] = args.eval_size
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
        cfg["qat"]["max_steps"] = args.max_steps
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    return cfg


def _write(cfg, name: str, obj) -> str:
    d = cfg["paths"]["results_dir"]
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    print("[write]", p)
    return p


def _load_for_eval(model_dir: str, precision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if precision == "bf16" else None  # int4 dirs carry their own config
    kwargs = dict(device_map="cuda")
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        if precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    return model, tok


def cmd_a(cfg, args):
    out = artifact_dir(cfg, "A_bf16", args.seed)
    log = V.train_method_a(cfg, args.seed, out, resume=args.resume)
    _write(cfg, f"A_train_seed{args.seed}.json", log)
    print(f"[A] seed={args.seed} loss={log['train_loss']:.4f} steps={log['global_step']} "
          f"secs={log['train_seconds']} -> {out}")


def cmd_b(cfg, args):
    a = artifact_dir(cfg, "A_bf16", args.seed)
    b = artifact_dir(cfg, "B_int4_ptq", args.seed)
    assert os.path.isdir(a), f"missing A merge: {a}"
    log = V.build_method_b(cfg, a, b)
    _write(cfg, f"B_build_seed{args.seed}.json", log)
    print(f"[B] seed={args.seed} size={log['size_gb']}GB -> {b}")


def cmd_c(cfg, args):
    a = artifact_dir(cfg, "A_bf16", args.seed)
    c = artifact_dir(cfg, "C_int4_qat", args.seed)
    assert os.path.isdir(a), f"missing A merge: {a}"
    log = V.train_method_c(cfg, a, c, args.seed, resume=args.resume)
    _write(cfg, f"C_train_seed{args.seed}.json", log)
    print(f"[C] seed={args.seed} loss={log['train_loss']:.4f} size={log['size_gb']}GB -> {c}")


def cmd_eval(cfg, args):
    prec = {"A_bf16": "bf16", "B_int4_ptq": "int4", "C_int4_qat": "int4"}[args.method]
    mdir = artifact_dir(cfg, args.method, args.seed)
    assert os.path.isdir(mdir), f"missing artifact: {mdir}"
    data = V.load_slices(cfg, split="final_holdout")   # P1-1: frozen holdout, eval-only
    model, tok = _load_for_eval(mdir, prec)
    res = V.eval_model_chat(cfg, model, tok, data["eval"], method=args.method,
                            seed=args.seed, model_dir=mdir, precision=prec,
                            split="final_holdout")
    _write(cfg, f"eval_{args.method}_seed{args.seed}.json", res)
    print(f"[eval] {args.method} seed={args.seed}: EM={res['exact_match']} F1={res['f1']} "
          f"ppl={res['perplexity']} vram={res['peak_vram_gb']}GB tok/s={res['tok_per_s']}")
    print("       samples:", res["sample_predictions"])


def cmd_selftest(cfg, args):
    import sys
    res = V.qat_scheme_selftest(gs=int(cfg["qat"]["group_size"]))
    _write(cfg, "qat_selftest.json", res)
    print(f"[selftest] ok={res['ok']} prepare_fires={res['prepare_fires']} "
          f"fq_layers={res['fake_quant_layers']} fake_err={res['fake_err_vs_orig']} "
          f"serve_err={res['serve_err_vs_orig']} ratio={res['fake_to_serve_ratio']} "
          f"same_family={res['same_int4_family']} convert_ok={res['convert_roundtrip_ok']}")
    if not res["ok"]:
        sys.exit(1)


def cmd_agg(cfg, args):
    d = cfg["paths"]["results_dir"]
    expected_seeds = set(cfg.get("seeds", []))
    expected_n = int(cfg["data"]["eval_size"])
    base = cfg["base_model"]["selected"]
    rows, skipped = [], []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("eval_") and fn.endswith(".json"):
            r = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            if (r.get("seed") in expected_seeds and r.get("n_eval") == expected_n
                    and r.get("base_model") == base):
                rows.append(r)
            else:
                skipped.append(fn)
    if skipped:
        print("[agg] skipped (seed/n_eval/base mismatch):", skipped)
    agg = V.aggregate_seeds(rows)
    _write(cfg, "three_way_table.json", {"base_model": base, "eval_size": expected_n,
                                         "seeds": sorted(expected_seeds), "per_seed": rows,
                                         "aggregate": agg})
    for m in ("A_bf16", "B_int4_ptq", "C_int4_qat"):
        if m in agg:
            a = agg[m]
            print(f"[agg] {m:12} F1={a['f1']['mean']}±{a['f1']['std']} "
                  f"EM={a['exact_match']['mean']}±{a['exact_match']['std']} "
                  f"ppl={a['perplexity']['mean']}±{a['perplexity']['std']} seeds={a['seeds']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("a", "b", "c", "eval", "agg", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=DEFAULT_CONFIG)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--method", default="A_bf16")
        p.add_argument("--subset", type=int, default=None)
        p.add_argument("--eval-size", type=int, default=None)
        p.add_argument("--max-steps", type=int, default=None)
        p.add_argument("--epochs", type=float, default=None)
    args = ap.parse_args()
    cfg = _apply_overrides(load_config(args.config), args)
    {"a": cmd_a, "b": cmd_b, "c": cmd_c, "eval": cmd_eval, "agg": cmd_agg,
     "selftest": cmd_selftest}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
