"""CLI entry point for the pdf_native_post_training workflow.

Usage:
    python -m workflows.pdf_native_post_training.cli --config <config.yaml>

Resolves a workflow config (docs, recorded/gold data, policy), assembles a run
bundle, runs the stage pipeline into ``runs/<run_id>/`` and writes a validated
``run_manifest.json`` + ``report.md``. Exits non-zero if any stage fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml  # noqa: E402

from pdf_qa.run_bundle import (  # noqa: E402
    RunBundle, environment_info, new_run_id, sha256_file, sha256_text, validate_manifest,
)
from workflows.pdf_native_post_training.pipeline import build_pipeline  # noqa: E402
from workflows.pdf_native_post_training.prompts import SYSTEM  # noqa: E402
from workflows.pdf_native_post_training.stages.harness import StageContext, run_pipeline  # noqa: E402

_MODE_TO_GENMODE = {
    "recorded_replay": "recorded_replay",
    "live_ollama": "live",
}


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_PKG_DIR, path)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    cfg["_config_path"] = os.path.abspath(config_path)
    cfg["_doc_paths"] = [_resolve(p) for p in cfg.get("docs", [])]
    if cfg.get("recorded_generations"):
        cfg["_recorded_path"] = _resolve(cfg["recorded_generations"])
    if cfg.get("gold_qa"):
        cfg["_gold_path"] = _resolve(cfg["gold_qa"])

    policy_name = cfg.get("policy", "public")
    policy_path = _resolve(os.path.join("configs", "policies", f"{policy_name}.yaml"))
    with open(policy_path, encoding="utf-8") as fh:
        cfg["_policy"] = yaml.safe_load(fh) or {}
    cfg["_policy_path"] = policy_path

    if cfg.get("train", {}).get("enabled") and not cfg.get("mode"):
        cfg["mode"] = "recorded_replay"
    return cfg


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PDF-native post-training workflow runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run-dir", default=None, help="explicit run dir (enables resume)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    mode = cfg.get("mode", "recorded_replay")
    runs_root = args.runs_root if os.path.isabs(args.runs_root) else os.path.join(_ROOT, args.runs_root)
    run_dir = args.run_dir or os.path.join(runs_root, new_run_id(cfg.get("run_prefix", "pdfnat")))
    run_id = os.path.basename(os.path.normpath(run_dir))

    bundle = RunBundle(
        run_id=run_id,
        command=f"pdf2llm run --config {os.path.basename(args.config)}",
        generation_mode=_MODE_TO_GENMODE.get(mode, "not_recorded"),
        base_dir=_ROOT,
    )
    bundle.out_base_dir = run_dir  # output paths relative to the run dir
    bundle.set_code(config_sha256=sha256_file(cfg["_config_path"]),
                    prompt_sha256=sha256_text(SYSTEM), cwd=_ROOT)
    bundle.environment = environment_info(["reportlab", "jsonschema", "yaml"])
    bundle.model = {"provider": cfg.get("provider", mode), "name": cfg.get("model", "demo-recorded"),
                    "revision": None}
    bundle.dataset = {"name": "public_finance_demo", "n_examples": None}
    bundle.seeds = list(cfg.get("seeds", [0]))
    for p in cfg["_doc_paths"]:
        bundle.add_input(p, role="source_pdf")
    if cfg.get("_gold_path"):
        bundle.add_input(cfg["_gold_path"], role="gold_qa")

    ctx = StageContext(config=cfg, run_dir=run_dir, base_dir=_ROOT, outputs={}, bundle=bundle)
    stages = build_pipeline(cfg)

    t0 = time.time()
    try:
        summary = run_pipeline(stages, ctx, resume=not args.no_resume)
    except Exception as exc:  # noqa: BLE001 - harness already recorded failure
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    bundle.elapsed_seconds = round(time.time() - t0, 3)

    manifest_path = bundle.write(run_dir)
    errs = validate_manifest(bundle.to_manifest())
    if errs:
        print(f"[FAIL] run manifest invalid: {errs}", file=sys.stderr)
        return 1

    rep = ctx.outputs.get("report", {})
    ev = rep.get("eval", {}).get("overall", {})
    print(f"[OK] {run_id}")
    print(f"  run_dir: {run_dir}")
    print(f"  manifest: {manifest_path}")
    print(f"  fingerprint: {bundle.reproducibility_fingerprint()[:16]}…")
    print(f"  evidence_address_integrity: {rep.get('evidence_address_integrity')}")
    print(f"  policy_quarantined: {rep.get('policy_quarantined')}  train_rows: {rep.get('train_rows_exported')}")
    print(f"  eval: EM {ev.get('em')} / F1 {ev.get('f1')} (n={ev.get('n')})")
    stage_statuses = {s["name"]: s["status"] for s in summary["stages"]}
    print(f"  stages: {stage_statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
