"""P1-9 tests — decision report (Pareto frontier, feasibility, cost, README parity)."""

from __future__ import annotations

import os
import re

from workflows.pdf_native_post_training import report as R

_HERE = os.path.dirname(os.path.abspath(__file__))
_WF = os.path.dirname(_HERE)
_DECISION_CFG = os.path.join(_WF, "configs", "decision_constraints.yaml")
_README = os.path.join(_WF, "README.md")


def _cfg():
    return R.load_decision_config(_DECISION_CFG)


# --- canonical committed example ------------------------------------------- #
def test_canonical_recommendation_is_deterministic():
    cfg = _cfg()
    dec = R.build_decision(cfg["candidates"], cfg["constraints"],
                           objectives=cfg["objectives"],
                           primary_objective=cfg["primary_objective"],
                           rate_card=cfg.get("rate_card"))
    # Memory-bound (<=8 GiB, <=1.0 F1 drop): A excluded by VRAM; among int4, QAT wins on F1.
    assert dec["recommendation"] == "C_int4_qat"
    assert dec["feasible"] == ["B_int4_ptq", "C_int4_qat"]


def test_pareto_frontier_drops_dominated_ptq():
    cfg = _cfg()
    dec = R.build_decision(cfg["candidates"], cfg["constraints"],
                           objectives=cfg["objectives"])
    # C dominates B (same size/vram/serving, higher F1) -> B off the frontier; A on it (best F1/TTFT).
    assert set(dec["pareto_frontier"]) == {"A_bf16", "C_int4_qat"}
    assert "B_int4_ptq" not in dec["pareto_frontier"]


def test_no_feasible_candidate():
    cfg = _cfg()
    tight = {"peak_vram_gb": {"max": 4}}  # nothing fits 4 GiB
    dec = R.build_decision(cfg["candidates"], tight, objectives=cfg["objectives"])
    assert dec["recommendation"] == R.NO_FEASIBLE
    assert dec["feasible"] == []


def test_constraint_on_not_measured_is_undecidable():
    cands = [
        {"id": "x", "f1": 90.0, "peak_vram_gb": 4.0, "ttft_p99_ms": "not_measured"},
        {"id": "y", "f1": 91.0, "peak_vram_gb": 4.0, "ttft_p99_ms": 100.0},
    ]
    constraints = {"ttft_p99_ms": {"max": 500}}
    by_id = {c["id"]: c for c in cands}
    vx = R.check_constraints(cands[0], constraints, by_id)
    vy = R.check_constraints(cands[1], constraints, by_id)
    assert vx["feasible"] is False and vx["undecidable"]      # cannot certify x
    assert vy["feasible"] is True and not vy["undecidable"]   # y is measured & passes
    dec = R.build_decision(cands, constraints,
                           objectives=[{"field": "f1", "direction": "max"}])
    assert dec["recommendation"] == "y"


def test_f1_drop_constraint_uses_baseline():
    cands = [
        {"id": "base", "f1": 95.0, "peak_vram_gb": 20.0},
        {"id": "small_ok", "f1": 94.2, "peak_vram_gb": 6.0},   # drop 0.8 <= 1.0
        {"id": "small_bad", "f1": 93.5, "peak_vram_gb": 6.0},  # drop 1.5 > 1.0
    ]
    constraints = {"peak_vram_gb": {"max": 8},
                   "f1": {"drop_max": 1.0, "drop_baseline": "base"}}
    dec = R.build_decision(cands, constraints,
                           objectives=[{"field": "f1", "direction": "max"}])
    assert dec["feasible"] == ["small_ok"]
    assert dec["recommendation"] == "small_ok"


# --- cost from a dated rate card ------------------------------------------- #
def test_cost_from_rate_card():
    usage = {"input_tokens": 1000, "output_tokens": 2000}
    card = {"source": "x", "as_of": "2025-01-01",
            "input_per_1k_usd": 0.5, "output_per_1k_usd": 1.5}
    assert R.compute_cost(usage, card) == 0.5 * 1 + 1.5 * 2  # 3.5


def test_cost_without_card_or_usage_is_not_measured():
    assert R.compute_cost(None, {"input_per_1k_usd": 1}) == "not_measured"
    assert R.compute_cost({"input_tokens": 10, "output_tokens": 10}, None) == "not_measured"


# --- report assembly + schema + 10 sections -------------------------------- #
def test_report_has_ten_sections_and_validates():
    from workflows.pdf_native_post_training import benchmark_pipeline as bp
    cfg = _cfg()
    report = R.build_report(cfg, sources=[{"path": "decision_constraints.yaml",
                                           "sha256": "0" * 64, "role": "decision_config"}])
    # schema-valid decision document
    assert bp.validate_metrics(report) == []
    md = R.render_report_md(report)
    for i in range(1, 11):
        assert f"## {i}." in md, f"missing section {i}"
    assert "recommendation: **C_int4_qat**" in md
    # real serving TTFT rendered (not masked as not_measured)
    assert "296.4" in md and "32.2" in md


def test_readme_recommendation_matches_recomputation():
    """Completion condition: the README's cited recommendation equals report.py's."""
    cfg = _cfg()
    dec = R.build_decision(cfg["candidates"], cfg["constraints"],
                           objectives=cfg["objectives"],
                           primary_objective=cfg["primary_objective"])
    with open(_README, encoding="utf-8") as fh:
        readme = fh.read()
    m = re.search(r"decision-report recommendation:\s*[\*`\s]*([A-Za-z0-9_]+)", readme)
    assert m, "README must cite the decision-report recommendation in a parseable form"
    assert m.group(1) == dec["recommendation"] == "C_int4_qat"
