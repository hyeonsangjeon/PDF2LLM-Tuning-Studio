"""P1-9: automatic decision report over the pipeline + quality + serving metrics.

Ties quality, model size, peak VRAM, TTFT/TPOT, throughput, goodput, error rate
and cost into one table, then computes the **Pareto frontier** and the candidates
that satisfy a config's constraints (e.g. ``peak_vram_gb <= 8``, ``f1_drop <= 1``,
``p95_ttft_ms <= 500``). If no candidate satisfies the constraints it returns
``no_feasible_candidate`` rather than inventing a recommendation.

Guarantees:

- every derived number comes from the raw metric records + this code — a human
  never hand-writes a derived value;
- a constraint over a ``not_measured`` field makes a candidate *undecidable*
  (it cannot be certified feasible), so CPU-only serving gaps never masquerade
  as a pass;
- cost is computed from a rate card carrying its own ``source`` + ``as_of``; with
  no rate card (or unmeasured usage) cost is ``not_measured``;
- ``report.md`` records the SHA-256 of every raw source it derived from.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

_NOT = ("not_measured", "not_applicable")
NO_FEASIBLE = "no_feasible_candidate"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


# --------------------------------------------------------------------------- #
# cost (computed from a dated rate card, never hard-coded)
# --------------------------------------------------------------------------- #
def compute_cost(usage: Optional[Dict[str, Any]], rate_card: Optional[Dict[str, Any]]) -> Any:
    if not rate_card or not usage:
        return "not_measured"
    it, ot = _num(usage.get("input_tokens")), _num(usage.get("output_tokens"))
    ip = _num(rate_card.get("input_per_1k_usd"))
    op = _num(rate_card.get("output_per_1k_usd"))
    if it is None or ot is None or ip is None or op is None:
        return "not_measured"
    return round(it / 1000.0 * ip + ot / 1000.0 * op, 6)


# --------------------------------------------------------------------------- #
# constraints -> feasibility (with undecidable on not_measured)
# --------------------------------------------------------------------------- #
def check_constraints(candidate: Dict[str, Any], constraints: Dict[str, Any],
                      by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return {feasible, unmet:[...], undecidable:[...]} for one candidate."""
    unmet: List[str] = []
    undecidable: List[str] = []
    for field, spec in (constraints or {}).items():
        val = candidate.get(field)
        num = _num(val)
        needs_value = any(k in spec for k in ("min", "max", "drop_max"))
        if needs_value and num is None:
            undecidable.append(f"{field}={val!r}")
            continue
        if "max" in spec and num is not None and num > float(spec["max"]) + 1e-9:
            unmet.append(f"{field} {num} > max {spec['max']}")
        if "min" in spec and num is not None and num < float(spec["min"]) - 1e-9:
            unmet.append(f"{field} {num} < min {spec['min']}")
        if "drop_max" in spec:
            base_id = spec.get("drop_baseline")
            base = by_id.get(base_id, {})
            base_num = _num(base.get(field))
            if base_num is None:
                undecidable.append(f"{field}.drop(baseline {base_id})")
            elif num is not None and (base_num - num) > float(spec["drop_max"]) + 1e-9:
                unmet.append(f"{field} drop {round(base_num - num, 4)} > {spec['drop_max']}")
    return {"feasible": not unmet and not undecidable,
            "unmet": unmet, "undecidable": undecidable}


# --------------------------------------------------------------------------- #
# Pareto frontier (not_measured objectives are incomparable)
# --------------------------------------------------------------------------- #
def _dominates(a: Dict[str, Any], b: Dict[str, Any], objectives: Sequence[Dict[str, str]]) -> bool:
    """a dominates b iff a is >= b on every measured objective and > on at least one.
    If any objective is not_measured for either, that objective is skipped; a can
    only dominate when at least one objective is strictly better and none worse."""
    strictly_better = False
    for obj in objectives:
        field, direction = obj["field"], obj.get("direction", "max")
        av, bv = _num(a.get(field)), _num(b.get(field))
        if av is None or bv is None:
            continue
        if direction == "min":
            av, bv = -av, -bv
        if av < bv - 1e-9:
            return False
        if av > bv + 1e-9:
            strictly_better = True
    return strictly_better


def pareto_frontier(candidates: Sequence[Dict[str, Any]],
                    objectives: Sequence[Dict[str, str]]) -> List[str]:
    frontier: List[str] = []
    for c in candidates:
        cid = c["id"]
        if not any(o["id"] != cid and _dominates(o, c, objectives) for o in candidates):
            frontier.append(cid)
    return frontier


# --------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------- #
def build_decision(candidates: Sequence[Dict[str, Any]], constraints: Dict[str, Any], *,
                   objectives: Optional[Sequence[Dict[str, str]]] = None,
                   primary_objective: str = "f1",
                   rate_card: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    objectives = objectives or [{"field": "f1", "direction": "max"},
                                {"field": "size_gb", "direction": "min"}]
    by_id = {c["id"]: c for c in candidates}

    rows: List[Dict[str, Any]] = []
    for c in candidates:
        verdict = check_constraints(c, constraints, by_id)
        cost = c.get("cost_usd")
        if cost is None:
            cost = compute_cost(c.get("provider_usage"), rate_card)
        rows.append({**c, "cost_usd": cost,
                     "feasible": verdict["feasible"],
                     "unmet": verdict["unmet"], "undecidable": verdict["undecidable"]})

    frontier = pareto_frontier(rows, objectives)
    feasible = [r["id"] for r in rows if r["feasible"]]

    prim_dir = next((o.get("direction", "max") for o in objectives
                     if o["field"] == primary_objective), "max")
    def _key(rid: str) -> float:
        v = _num(by_id[rid].get(primary_objective))
        v = v if v is not None else float("-inf")
        return -v if prim_dir == "min" else v

    recommendation = max(feasible, key=_key) if feasible else NO_FEASIBLE
    return {
        "candidates": rows,
        "constraints": constraints,
        "objectives": list(objectives),
        "primary_objective": primary_objective,
        "pareto_frontier": frontier,
        "feasible": feasible,
        "recommendation": recommendation,
    }


# --------------------------------------------------------------------------- #
# assembly + markdown
# --------------------------------------------------------------------------- #
def build_report(decision_config: Dict[str, Any], *,
                 pipeline_metrics: Optional[Dict[str, Any]] = None,
                 manifest: Optional[Dict[str, Any]] = None,
                 sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    decision = build_decision(
        decision_config.get("candidates", []),
        decision_config.get("constraints", {}),
        objectives=decision_config.get("objectives"),
        primary_objective=decision_config.get("primary_objective", "f1"),
        rate_card=decision_config.get("rate_card"))
    doc = {
        "schema_version": "pdf2llm-metrics/1",
        "kind": "decision",
        "generated_at": _utc_now(),
        "generated_by": "workflows.pdf_native_post_training.report",
        "sources": sources or [],
        "rate_card": decision_config.get("rate_card"),
        "acceptance": decision_config.get("acceptance", {}),
        "status": decision_config.get("status", "replay"),
        "limitations": decision_config.get("limitations", []),
        "manifest_excerpt": _manifest_excerpt(manifest),
        "decision": {k: decision[k] for k in
                     ("constraints", "objectives", "primary_objective",
                      "pareto_frontier", "feasible", "recommendation")},
        "candidates": decision["candidates"],
        "recommendation": decision["recommendation"],
    }
    pm = (pipeline_metrics or {}).get("pipeline")
    if pm is not None:
        doc["pipeline"] = pm
    return doc


def _manifest_excerpt(manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not manifest:
        return {}
    return {"code": manifest.get("code"), "model": manifest.get("model"),
            "dataset": manifest.get("dataset"), "container": manifest.get("container"),
            "artifact_set_hash": manifest.get("artifact_set_hash"),
            "reproducibility_fingerprint": manifest.get("reproducibility_fingerprint")}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def render_report_md(report: Dict[str, Any]) -> str:
    p = report.get("pipeline") or {}
    dec = report.get("decision") or {}
    rows = report.get("candidates") or []
    mx = report.get("manifest_excerpt") or {}
    L: List[str] = ["# PDF-native pipeline — decision report", "",
                    f"- generated_at: {report.get('generated_at')}  ·  status: **{report.get('status')}**",
                    f"- recommendation: **{report.get('recommendation')}**", ""]

    # 1. inputs
    L += ["## 1. 입력 문서 (source · SHA256 · classification · license)"]
    for s in report.get("sources", []) or []:
        L.append(f"- `{s.get('path')}` — sha256 `{(s.get('sha256') or 'none')[:16]}…` "
                 f"({s.get('role', 'source')})")
    if not report.get("sources"):
        L.append("- not_measured")
    L.append("")

    # 2. raw -> accepted -> rejected waterfall
    qa = p.get("qa", {})
    L += ["## 2. raw → accepted → rejected waterfall",
          f"- raw **{qa.get('raw', 'not_measured')}** → accepted **{qa.get('accepted', 'not_measured')}** "
          f"→ rejected **{qa.get('rejected', 'not_measured')}** (yield {qa.get('yield', 'not_measured')})",
          f"- reject reasons: {p.get('reject_reasons') or 'none'}", ""]

    # 3. evidence linkage + policy gate
    L += ["## 3. 근거 연결률 · policy gate",
          f"- evidence pass rate: **{p.get('evidence_pass_rate', 'not_measured')}**",
          f"- figure-caption linkage: {p.get('figure_caption_linkage_rate', 'not_measured')}",
          f"- provider usage (egress): {p.get('provider_usage', 'not_measured')}", ""]

    # 4. split / training config / artifact
    L += ["## 4. split · training config · loss · artifact",
          f"- code: `{(mx.get('code') or {}).get('git_sha', 'not_measured')}`  ·  "
          f"artifact_set_hash: `{(mx.get('artifact_set_hash') or 'not_measured')}`",
          f"- artifact bytes: {p.get('artifact_bytes', 'not_measured')}", ""]

    # 5. quality table
    L += ["## 5. 품질 (Base/SFT/PTQ/QAT …) · category별",
          "| candidate | f1 | em | size_gb | pareto | feasible |",
          "|---|---|---|---|---|---|"]
    frontier = set(dec.get("pareto_frontier", []))
    for r in rows:
        L.append(f"| {r['id']} | {_fmt(r.get('f1', 'n/a'))} | {_fmt(r.get('em', 'n/a'))} | "
                 f"{_fmt(r.get('size_gb', 'n/a'))} | {'✓' if r['id'] in frontier else ''} | "
                 f"{'✓' if r.get('feasible') else '✗'} |")
    L.append("")

    # 6. serving
    L += ["## 6. serving latency · throughput · VRAM · error rate",
          "| candidate | ttft_ms | tpot_ms | throughput_tok_s | peak_vram_gb | goodput | error_rate |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        ttft = r.get("ttft_p50_ms", r.get("ttft_ms", "not_measured"))
        L.append(f"| {r['id']} | {_fmt(ttft)} | "
                 f"{_fmt(r.get('tpot_ms', 'not_measured'))} | {_fmt(r.get('throughput_tok_s', 'not_measured'))} | "
                 f"{_fmt(r.get('peak_vram_gb', 'not_measured'))} | {_fmt(r.get('goodput', 'not_measured'))} | "
                 f"{_fmt(r.get('error_rate', 'not_measured'))} |")
    L.append("")

    # 7. review queue + representative failures
    L += ["## 7. review queue · 대표 실패 사례",
          f"- manual review minutes: {p.get('manual_review_minutes', 'not_measured')}",
          f"- reject reasons: {p.get('reject_reasons') or 'none'}", ""]

    # 8. revisions + hashes
    L += ["## 8. code / model / data / container revision · artifact hash",
          f"- model: {mx.get('model', 'not_measured')}",
          f"- dataset: {mx.get('dataset', 'not_measured')}",
          f"- container: {mx.get('container', 'not_measured')}",
          f"- reproducibility_fingerprint: `{(mx.get('reproducibility_fingerprint') or 'not_measured')}`", ""]

    # 9. acceptance gates PASS/FAIL
    L += ["## 9. acceptance gate (PASS/FAIL)"]
    for r in rows:
        status = "PASS" if r.get("feasible") else "FAIL"
        why = ("; ".join(r.get("unmet", []) + [f"undecidable {u}" for u in r.get("undecidable", [])])
               or "all constraints met")
        L.append(f"- {r['id']}: **{status}** — {why}")
    L += [f"- **decision: {report.get('recommendation')}**", ""]

    # 10. status + limitations
    L += ["## 10. 상태 · limitations",
          f"- status: **{report.get('status')}** (replay/live/historical/planned)"]
    for lim in report.get("limitations", []) or []:
        L.append(f"- limitation: {lim}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# IO + CLI
# --------------------------------------------------------------------------- #
def load_decision_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(fh)
        return json.load(fh)


def _sha(path: str) -> Optional[str]:
    from pdf_qa.run_bundle import sha256_file
    try:
        return sha256_file(path)
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="workflows.pdf_native_post_training.report",
        description="Automatic decision report (Pareto + constraints) over pipeline/quality/serving metrics (P1-9).")
    ap.add_argument("--decision-config", required=True,
                    help="YAML/JSON with candidates + constraints + objectives + rate_card")
    ap.add_argument("--pipeline-metrics", default=None, help="pipeline metrics JSON (benchmark_pipeline)")
    ap.add_argument("--manifest", default=None, help="run_manifest.json for revisions/hashes")
    ap.add_argument("--out-dir", required=True, help="report output directory (report.json + report.md)")
    ap.add_argument("--expect-recommendation", default=None,
                    help="assert the recommendation equals this (exit non-zero otherwise)")
    args = ap.parse_args(argv)

    dcfg = load_decision_config(args.decision_config)
    pm = json.load(open(args.pipeline_metrics, encoding="utf-8")) if args.pipeline_metrics else None
    man = json.load(open(args.manifest, encoding="utf-8")) if args.manifest else None

    sources = [{"path": os.path.basename(args.decision_config), "sha256": _sha(args.decision_config),
                "role": "decision_config"}]
    if args.pipeline_metrics:
        sources.append({"path": os.path.basename(args.pipeline_metrics),
                        "sha256": _sha(args.pipeline_metrics), "role": "pipeline_metrics"})
    if args.manifest:
        sources.append({"path": "run_manifest.json", "sha256": _sha(args.manifest), "role": "run_manifest"})

    report = build_report(dcfg, pipeline_metrics=pm, manifest=man, sources=sources)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_report_md(report))

    print(f"[report] {args.out_dir}/report.md")
    print(f"[report] recommendation = {report['recommendation']}  "
          f"feasible={report['decision']['feasible']}  pareto={report['decision']['pareto_frontier']}")
    if args.expect_recommendation is not None and report["recommendation"] != args.expect_recommendation:
        print(f"[report] MISMATCH: expected {args.expect_recommendation!r}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
