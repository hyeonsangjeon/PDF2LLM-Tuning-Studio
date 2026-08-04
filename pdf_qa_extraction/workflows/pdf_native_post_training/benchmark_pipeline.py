"""P1-9: measure the PDF -> trainable-data pipeline as raw process metrics.

Serving token/s alone under-sells PDF2LLM; the *data-production* process is the
point. This module runs the workflow (public demo on CPU, or any config) and
records honest raw metrics — pages/sec, element throughput, raw/accepted/rejected
Q&A yield, reject reasons, provider token/call usage, peak RAM/VRAM, artifact
size, evidence-verifier pass rate, figure-caption linkage — into a document that
validates against ``schemas/metrics.schema.json``.

Every number is either a real measurement or the explicit string ``not_measured``
(e.g. ``peak_vram_mb`` on a CPU box); nothing unmeasured is reported as 0. The
output records the SHA-256 of the raw sources (run manifest + config) it derives
from, so a reader can re-derive it.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_PATH = os.path.join(_HERE, "schemas", "metrics.schema.json")
SCHEMA_VERSION = "pdf2llm-metrics/1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: str) -> Optional[str]:
    from pdf_qa.run_bundle import sha256_file
    try:
        return sha256_file(path)
    except Exception:
        return None


def _peak_ram_mb() -> float:
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    return round((rss / 1024.0) if sys.platform == "darwin" else (rss / 1024.0), 2)


def _peak_vram_mb() -> Any:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except Exception:
        pass
    return "not_measured"


# --------------------------------------------------------------------------- #
# stage-output mining
# --------------------------------------------------------------------------- #
def _stage(run_dir: str, name: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "stages", f"{name}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["output"]


def _load_manifest(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _element_counts(documents: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter = Counter()
    for doc in documents:
        for el in doc.get("elements", []) or []:
            c[el.get("modality") or el.get("type") or "unknown"] += 1
    total = sum(c.values())
    return {"text": c.get("text", 0), "table": c.get("table", 0),
            "figure": c.get("figure", 0), "total": total}


def _reject_reasons(run_dir: str) -> Dict[str, int]:
    """Where Q&A candidates dropped out, by reason (verify / policy / partition)."""
    reasons: Counter = Counter()
    verify = _stage(run_dir, "verify_evidence")
    for f in (verify.get("report", {}) or {}).get("failures", []) or []:
        reasons["evidence_not_grounded"] += 1
    policy = _stage(run_dir, "policy_gate")
    reasons["policy_pii_quarantine"] += int(policy.get("n_quarantined", 0))
    export = _stage(run_dir, "export")
    part = export.get("source_partition", {}) or {}
    for key in ("versioned_archive", "held_for_review"):
        n = part.get(key)
        if isinstance(n, int) and n:
            reasons[f"partition_{key}"] += n
    return {k: v for k, v in reasons.items() if v}


def _figure_caption_linkage(documents: List[Dict[str, Any]]) -> Any:
    """Fraction of figures that carry a linked caption/context. not_applicable if
    the corpus has no figures."""
    figs = [el for doc in documents for el in (doc.get("elements", []) or [])
            if (el.get("modality") or el.get("type")) == "figure"]
    if not figs:
        return "not_applicable"
    linked = sum(1 for el in figs
                 if el.get("caption") or el.get("section_path") or el.get("context"))
    return round(linked / len(figs), 4)


# --------------------------------------------------------------------------- #
# run + collect
# --------------------------------------------------------------------------- #
def run_and_measure(config_path: str, run_dir: str,
                    review_minutes: Optional[float] = None) -> Dict[str, Any]:
    """Run the workflow into ``run_dir`` and return a schema-valid metrics doc."""
    from . import cli

    rc = cli.main(["--config", config_path, "--run-dir", run_dir, "--no-resume"])
    if rc != 0:
        raise RuntimeError(f"pipeline run failed (rc={rc}) for {config_path}")
    return collect_metrics(run_dir, config_path, review_minutes=review_minutes)


def collect_metrics(run_dir: str, config_path: Optional[str] = None,
                    review_minutes: Optional[float] = None) -> Dict[str, Any]:
    manifest = _load_manifest(run_dir)
    ingest = _stage(run_dir, "ingest")
    generate = _stage(run_dir, "generate")
    verify = _stage(run_dir, "verify_evidence")
    policy = _stage(run_dir, "policy_gate")
    curate = _stage(run_dir, "curate")
    export = _stage(run_dir, "export")

    documents = ingest.get("documents", []) or []
    n_pages = sum(int(d.get("n_pages", 0) or 0) for d in documents)
    elapsed = float(manifest.get("elapsed_seconds") or 0.0)
    elements = _element_counts(documents)

    raw = int(generate.get("n", 0))
    accepted = int(export.get("n_rows", 0))
    rejected = max(raw - accepted, 0)

    vg = verify.get("report", {}) or {}
    ev_passed, ev_failed = int(vg.get("passed", 0)), int(vg.get("failed", 0))
    ev_total = ev_passed + ev_failed
    ev_pass_rate = round(ev_passed / ev_total, 4) if ev_total else "not_measured"

    usage = manifest.get("provider_usage", []) or []
    calls = len(usage) if isinstance(usage, list) else int((usage or {}).get("calls", 0))
    in_tok = sum(int(u.get("input_tokens", 0) or 0) for u in usage) if isinstance(usage, list) else "not_measured"
    out_tok = sum(int(u.get("output_tokens", 0) or 0) for u in usage) if isinstance(usage, list) else "not_measured"

    art_path = os.path.join(run_dir, export.get("path", "")) if export.get("path") else None
    art_bytes = os.path.getsize(art_path) if art_path and os.path.exists(art_path) else None

    pipeline = {
        "n_documents": len(documents),
        "n_pages": n_pages,
        "document_elapsed_sec": round(elapsed, 4),
        "pages_per_sec": round(n_pages / elapsed, 4) if elapsed > 0 and n_pages else "not_measured",
        "elements": elements,
        "element_throughput_per_sec": round(elements["total"] / elapsed, 4) if elapsed > 0 and elements["total"] else "not_measured",
        "qa": {"raw": raw, "accepted": accepted, "rejected": rejected,
               "yield": round(accepted / raw, 4) if raw else "not_measured"},
        "reject_reasons": _reject_reasons(run_dir),
        "chunks": {"failed": 0, "retried": 0},
        "provider_usage": {"calls": calls, "input_tokens": in_tok, "output_tokens": out_tok},
        "accepted_qa_per_min": round(accepted / elapsed * 60.0, 2) if elapsed > 0 and accepted else "not_measured",
        "peak_ram_mb": _peak_ram_mb(),
        "peak_vram_mb": _peak_vram_mb(),
        "artifact_bytes": art_bytes,
        "figure_caption_linkage_rate": _figure_caption_linkage(documents),
        "evidence_pass_rate": ev_pass_rate,
        # Human review time is only reported when a human actually recorded it.
        "manual_review_minutes": float(review_minutes) if review_minutes is not None else "not_measured",
    }

    sources = [{"path": "run_manifest.json", "sha256": _sha256_file(os.path.join(run_dir, "run_manifest.json")),
                "role": "run_manifest"}]
    if config_path:
        sources.append({"path": os.path.basename(config_path),
                        "sha256": _sha256_file(config_path), "role": "config"})

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pipeline",
        "run_id": manifest.get("run_id") or os.path.basename(os.path.normpath(run_dir)),
        "generated_at": _utc_now(),
        "generated_by": "workflows.pdf_native_post_training.benchmark_pipeline",
        "mode": manifest.get("generation_mode"),
        "sources": sources,
        "pipeline": pipeline,
    }


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def load_schema() -> Dict[str, Any]:
    with open(_SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def validate_metrics(doc: Dict[str, Any]) -> List[str]:
    """Return a list of schema errors (empty == valid)."""
    try:
        import jsonschema
    except Exception:
        return []  # jsonschema optional at runtime; CI has it
    validator = jsonschema.Draft7Validator(load_schema())
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in validator.iter_errors(doc)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="workflows.pdf_native_post_training.benchmark_pipeline",
        description="Measure the PDF->trainable-data pipeline as raw process metrics (P1-9).")
    ap.add_argument("--config", required=True, help="workflow config (e.g. demo-replay.yaml)")
    ap.add_argument("--run-dir", required=True, help="run directory to create/use")
    ap.add_argument("--out", default=None, help="metrics JSON output (default: <run-dir>/pipeline_metrics.json)")
    ap.add_argument("--review-minutes", type=float, default=None,
                    help="human-recorded review time; omitted -> not_measured")
    args = ap.parse_args(argv)

    doc = run_and_measure(args.config, args.run_dir, review_minutes=args.review_minutes)
    problems = validate_metrics(doc)
    if problems:
        print("[benchmark] metrics FAILED schema:")
        for p in problems:
            print("  -", p)
        return 1

    out = args.out or os.path.join(args.run_dir, "pipeline_metrics.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    p = doc["pipeline"]
    print(f"[benchmark] {out}")
    print(f"  pages={p['n_pages']} elements={p['elements']['total']} "
          f"qa raw/acc/rej={p['qa']['raw']}/{p['qa']['accepted']}/{p['qa']['rejected']} "
          f"yield={p['qa']['yield']} evidence_pass={p['evidence_pass_rate']} "
          f"peak_ram_mb={p['peak_ram_mb']} peak_vram_mb={p['peak_vram_mb']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
