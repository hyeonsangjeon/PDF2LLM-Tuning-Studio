"""P2-3: asset license-ledger gate.

These exercise ``scripts/check_asset_ledger.py`` against the real ledger and
against crafted trees, proving that (a) the current repo passes ``--check``,
(b) an unlisted committed asset fails, (c) a stale ledger entry fails, (d)
schema violations fail, and (e) ``--release`` blocks an ``unresolved`` asset
while ``--check`` does not. No GPU or network required.
"""

import copy
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import check_asset_ledger as A  # noqa: E402


def _real_ledger():
    return A._load_ledger()


def test_real_ledger_check_passes():
    # The committed tree must pass the CI gate today.
    assert A.main(["--check"]) == 0


def test_audit_reports_no_errors_for_real_tree():
    result = A.audit(_real_ledger())
    assert result["errors"] == [], result["errors"]
    # Every committed asset is mapped to exactly one entry.
    assert result["n_committed"] == len(result["coverage"]) > 0


def test_unlisted_committed_asset_fails(monkeypatch):
    real = A._committed_assets()
    monkeypatch.setattr(
        A, "_committed_assets",
        lambda: sorted(real + ["pdf_qa_extraction/data/UNLISTED_asset.png"]),
    )
    result = A.audit(_real_ledger())
    assert any("not in ledger" in e and "UNLISTED_asset" in e
               for e in result["errors"])
    assert A.main(["--check"]) == 1


def test_stale_ledger_entry_fails():
    ledger = copy.deepcopy(_real_ledger())
    ledger["assets"].append({
        "id": "ghost",
        "path": "pdf_qa_extraction/data/does_not_exist.pdf",
        "kind": "sample_pdf",
        "license": "MIT",
        "source": "n/a",
        "redistribution": "ok",
    })
    result = A.audit(ledger)
    assert any("stale ledger entry" in e and "ghost" in e for e in result["errors"])


def test_invalid_redistribution_value_fails():
    ledger = copy.deepcopy(_real_ledger())
    ledger["assets"][0]["redistribution"] = "totally-fine"
    errors = A._validate_schema(ledger)
    assert any("invalid redistribution" in e for e in errors)


def test_missing_required_field_fails():
    ledger = copy.deepcopy(_real_ledger())
    ledger["assets"][0].pop("license")
    errors = A._validate_schema(ledger)
    assert any("missing required field 'license'" in e for e in errors)


def test_release_blocks_unresolved_but_check_does_not():
    # There is at least one unresolved committed asset (fsi_data.pdf) today.
    result = A.audit(_real_ledger())
    assert result["unresolved"], "expected >=1 unresolved committed asset"
    # --check tolerates it (documents the known-unknown); --release blocks it.
    assert A.main(["--check"]) == 0
    assert A.main(["--release"]) == 1


def test_duplicate_entry_id_fails():
    ledger = copy.deepcopy(_real_ledger())
    dup = copy.deepcopy(ledger["assets"][0])
    ledger["assets"].append(dup)
    errors = A._validate_schema(ledger)
    assert any("duplicate entry id" in e for e in errors)
