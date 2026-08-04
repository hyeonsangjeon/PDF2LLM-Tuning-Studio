"""Tests for the evidence index (spec P0-11).

Ensures the claim ledger stays honest: every number verifies against raw JSON +
README, EVIDENCE.md never drifts, a corrupted claim fails, and a `planned` feature
cannot carry a measured number. Import-light (PyYAML only, already a workflow dep).
"""
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import build_evidence_index as B  # noqa: E402


def test_real_ledger_check_passes():
    rc = B.main(["--check"])
    assert rc == 0


def test_evidence_md_is_current():
    ledger = B.load_ledger(B.DEFAULT_LEDGER)
    want = B.render_evidence_md(ledger)
    with open(B.DEFAULT_EVIDENCE_MD, encoding="utf-8") as fh:
        assert fh.read() == want, "docs/EVIDENCE.md is stale — run --emit"


def test_all_claims_verify_individually():
    ledger = B.load_ledger(B.DEFAULT_LEDGER)
    ok, errors = B.verify_all(ledger, B._ROOT)
    assert ok, errors


def test_resolve_pointer_dict_list_and_escaping():
    doc = {"a": {"b": [10, {"c": 3}]}, "x/y": 7, "~z": 9}
    assert B.resolve_pointer(doc, "/a/b/0") == 10
    assert B.resolve_pointer(doc, "/a/b/1/c") == 3
    assert B.resolve_pointer(doc, "/x~1y") == 7   # ~1 -> /
    assert B.resolve_pointer(doc, "/~0z") == 9    # ~0 -> ~


def test_corrupted_number_fails():
    ledger = B.load_ledger(B.DEFAULT_LEDGER)
    for c in ledger["claims"]:
        if c["id"] == "a_bf16_f1":
            c["expected"] = "99.99"
    ok, errors = B.verify_all(ledger, B._ROOT)
    assert not ok
    assert any("a_bf16_f1" in e for e in errors)


def test_planned_with_number_is_rejected():
    ledger = B.load_ledger(B.DEFAULT_LEDGER)
    for c in ledger["claims"]:
        if c["id"] == "grpo_rl":
            c["expected"] = "42.0"
            c["pointer"] = "/x"
    ok, errors = B.verify_all(ledger, B._ROOT)
    assert not ok
    assert any("planned feature must NOT carry" in e for e in errors)


def test_bad_status_is_rejected():
    errs = B.verify_claim({"id": "z", "capability": "c", "status": "totally_done"}, B._ROOT)
    assert any("not in" in e for e in errs)


def test_string_expected_substring_match():
    # Base-select winner: JSON value is "Qwen/Qwen3-8B", expected "Qwen3-8B" (substring).
    ledger = B.load_ledger(B.DEFAULT_LEDGER)
    winner = next(c for c in ledger["claims"] if c["id"] == "base_select_winner")
    assert B.verify_claim(winner, B._ROOT) == []


def test_allowed_status_set_matches_spec():
    assert set(B.ALLOWED_STATUS) == {
        "ci_verified", "recorded_hardware_run",
        "historical_not_reproduced", "planned"}
