"""Trust / correctness regression guards (P0-0, P0-5).

scan-secrets: allow-file (positive-test fixtures contain pattern shapes)

These run on plain CPU CI and fail if the removed blockers ever reappear:
* personal / address PII or secret *shapes* in the public fine_tuning tree,
* the ``eval_dataset=dataset`` train/eval leakage code smell,
* raw (unfiltered) rows written straight to the training export.

The scanner matches generic shapes only; no real name is hardcoded here.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, "pdf_qa_extraction", "scripts")
sys.path.insert(0, SCRIPTS)

import scan_secrets  # noqa: E402


def test_fine_tuning_tree_has_no_pii_or_leak():
    ft = os.path.join(REPO, "fine_tuning")
    findings = scan_secrets.scan([ft])
    assert findings == [], "trust regression: " + "; ".join(
        f"{f.path}:{f.line}[{f.kind}]" for f in findings
    )


def test_scanner_catches_planted_leak(tmp_path):
    p = tmp_path / "leaky.py"
    p.write_text("trainer = SFT(train_dataset=ds, eval_dataset=dataset)\n", encoding="utf-8")
    kinds = {f.kind for f in scan_secrets.scan([str(tmp_path)])}
    assert "train_eval_leak" in kinds


def test_scanner_catches_email_but_allows_example_domain(tmp_path):
    (tmp_path / "real.txt").write_text("contact me at john.doe@gmail.com\n", encoding="utf-8")
    (tmp_path / "synthetic.txt").write_text("reviewer@example.com is a canary\n", encoding="utf-8")
    kinds = [(f.path, f.kind) for f in scan_secrets.scan([str(tmp_path)])]
    assert any(k == "email" and p.endswith("real.txt") for p, k in kinds)
    assert not any(p.endswith("synthetic.txt") for p, _ in kinds)


def test_training_export_contains_only_clean_rows():
    """The committed train_data.jsonl must have no PII-shaped content."""
    jl = os.path.join(REPO, "fine_tuning", "data", "train_data.jsonl")
    rows = [json.loads(l) for l in open(jl, encoding="utf-8") if l.strip()]
    assert rows, "train_data.jsonl unexpectedly empty"
    # every row is well-formed instruction/output
    for r in rows:
        assert r.get("instruction") and r.get("output")
    findings = scan_secrets.scan([jl])
    assert findings == [], [f"{f.line}:{f.kind}" for f in findings]
