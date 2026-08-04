"""Tests for the base-selection harness (spec P0-4 completion conditions).

Covered: result-JSON schema, deterministic candidate ordering + seed fixing,
gated->fallback resolution, and the read-only ``--check-historical`` contract
(the committed ``results/base_select.json`` SHA-256 must not change). These stay
import-light — no torch / model downloads.
"""
import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantization import bench_baseselect as B  # noqa: E402
from quantization.data_korquad import DEFAULT_CONFIG, load_config  # noqa: E402


# --------------------------------------------------------------------------- #
# Result JSON schema
# --------------------------------------------------------------------------- #
def test_historical_results_pass_schema():
    with open(B.HISTORICAL_RESULTS, encoding="utf-8") as fh:
        entries = json.load(fh)
    B.validate_results(entries)  # must not raise
    assert len(entries) >= 2


def test_validate_entry_rejects_missing_keys():
    with pytest.raises(ValueError):
        B.validate_entry({"candidate": "x", "family": "y"})
    with pytest.raises(ValueError):
        B.validate_entry({"candidate": "x", "family": "y", "n_eval": 1,
                          "zeroshot": {"exact_match": 1.0}, "fewshot": {}})


# --------------------------------------------------------------------------- #
# Deterministic ordering + winner (candidate ordering must be reproducible)
# --------------------------------------------------------------------------- #
def test_ranking_is_by_zeroshot_f1_desc_and_deterministic():
    entries = [
        {"candidate": "b", "zeroshot": {"f1": 80.0}, "fewshot": {"f1": 1.0}},
        {"candidate": "a", "zeroshot": {"f1": 90.0}, "fewshot": {"f1": 1.0}},
        {"candidate": "c", "zeroshot": {"f1": 90.0}, "fewshot": {"f1": 1.0}},
    ]
    ranked = [e["candidate"] for e in B.rank_candidates(entries)]
    # 90.0 tie broken by candidate id ('a' < 'c'), then 80.0
    assert ranked == ["a", "c", "b"]
    # stable across repeated calls
    assert ranked == [e["candidate"] for e in B.rank_candidates(entries)]


def test_historical_winner_is_qwen3_8b():
    with open(B.HISTORICAL_RESULTS, encoding="utf-8") as fh:
        entries = json.load(fh)
    assert B.select_winner(entries) == "Qwen/Qwen3-8B"


def test_winner_matches_config_selected():
    cfg = load_config(DEFAULT_CONFIG)
    with open(B.HISTORICAL_RESULTS, encoding="utf-8") as fh:
        entries = json.load(fh)
    assert B.select_winner(entries) == cfg["base_model"]["selected"]


# --------------------------------------------------------------------------- #
# Seed fixing + provenance
# --------------------------------------------------------------------------- #
def test_provenance_records_fixed_seed_from_config():
    cfg = load_config(DEFAULT_CONFIG)
    prov = B.build_provenance(cfg, mode="gpu", fewshot_k=2,
                              config_path=DEFAULT_CONFIG, reproduced=True)
    assert prov["data_seed"] == int(cfg["data"]["seed"])
    assert prov["generated_by"] == "quantization.bench_baseselect"
    assert prov["reproduced"] is True
    assert prov["config_sha256"] and len(prov["config_sha256"]) == 64


# --------------------------------------------------------------------------- #
# Gated -> ungated fallback resolution
# --------------------------------------------------------------------------- #
def test_resolve_candidates_gating_deterministic():
    cfg = load_config(DEFAULT_CONFIG)
    no_token = B.resolve_candidates(cfg, token=False)
    with_token = B.resolve_candidates(cfg, token=True)
    assert len(no_token) == len(cfg["base_model"]["candidates"])
    # deterministic
    assert no_token == B.resolve_candidates(cfg, token=False)
    # the gated Llama candidate falls back to its ungated model without a token
    gated = [c for c in cfg["base_model"]["candidates"] if c.get("gated")]
    if gated and gated[0].get("fallback"):
        fb = next(c for c in no_token if c["gated_fallback_from"])
        assert fb["model_id"] == gated[0]["fallback"]
        # with a token present it stays on the gated model
        assert all(c["gated_fallback_from"] is None for c in with_token)


# --------------------------------------------------------------------------- #
# --check-historical is strictly read-only
# --------------------------------------------------------------------------- #
def test_inspect_historical_is_read_only():
    before = B.sha256_file(B.HISTORICAL_RESULTS)
    report = B.inspect_historical()
    assert report["exists"] is True
    assert report["reproduced_by_this_cli"] is False
    assert report["status"] == "historical_not_reproduced"
    assert report["winner"] == "Qwen/Qwen3-8B"
    assert report["sha256_unchanged"] is True
    assert B.sha256_file(B.HISTORICAL_RESULTS) == before  # untouched


def test_cli_check_historical_returns_zero_and_untouched(capsys):
    before = B.sha256_file(B.HISTORICAL_RESULTS)
    rc = B.main(["--check-historical"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["winner"] == "Qwen/Qwen3-8B"
    assert B.sha256_file(B.HISTORICAL_RESULTS) == before


def test_cli_list_candidates_returns_zero(capsys):
    rc = B.main(["--list-candidates"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["selected_in_config"] == "Qwen/Qwen3-8B"
    assert len(out["candidates"]) >= 3


def test_cli_refuses_to_overwrite_historical():
    with pytest.raises(SystemExit):
        B.main(["--out", B.HISTORICAL_RESULTS])


def test_default_output_is_under_runs():
    p = B.default_output_path("myrun")
    assert p.replace(os.sep, "/") == "runs/myrun/quantization/base_select.json"
