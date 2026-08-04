"""P1-6: tests for stable/mutable fact separation and source selection."""

import json
import os
import sys

import pytest

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from workflows.pdf_native_post_training import source_selection as S  # noqa: E402

_FIX = os.path.join(os.path.dirname(__file__), "..", "public_finance_demo",
                    "versioned_facts.jsonl")
_SCHEMA = os.path.join(_PKG, "pdf_qa", "schemas", "qa_with_evidence.schema.json")


@pytest.fixture(scope="module")
def records():
    return S.load_records(_FIX)


@pytest.fixture(scope="module")
def groups(records):
    return S.group_by_fact(records)


# --------------------------------------------------------------------------- #
# select latest valid source
# --------------------------------------------------------------------------- #
def test_selects_latest_active_over_superseded(groups):
    d = S.select_source(groups["policy_rate"])
    assert not d.abstain
    assert d.selected["document_version"] == "v2"
    assert "3.00%" in d.answer


def test_effective_date_window_picks_period_correct_source(groups):
    atm = groups["atm_fee"]
    assert "1,000원" in S.select_source(atm, as_of="2023-06-01").answer
    assert "1,200원" in S.select_source(atm, as_of="2024-06-01").answer
    assert "1,200원" in S.select_source(atm, as_of=None).answer  # latest by default


def test_stable_fact_survives_only_stale_source(groups):
    d = S.select_source(groups["headquarters"])
    assert not d.abstain
    assert d.reason == "selected_stable_from_stale"
    assert "서울" in d.answer


# --------------------------------------------------------------------------- #
# never confidently answer an outdated / ambiguous value -> abstain
# --------------------------------------------------------------------------- #
def test_conflicting_active_sources_without_order_abstain(groups):
    d = S.select_source(groups["credit_limit"])
    assert d.abstain and d.reason == "conflict_no_order"
    assert d.answer is None


def test_revoked_only_source_abstains(groups):
    d = S.select_source(groups["annual_fee"])
    assert d.abstain
    assert d.answer is None


def test_only_stale_mutable_source_abstains():
    stale_only = [{
        "qa_id": "x", "question": "금리?", "answer": "연 9.99%", "answerable": True,
        "fact_volatility": "mutable", "source_status": "stale",
        "document_version": "v1", "fact_key": "rate_x",
    }]
    d = S.select_source(stale_only)
    assert d.abstain and d.reason == "only_stale_mutable"
    assert d.answer is None


def test_no_candidates_abstains():
    d = S.select_source([])
    assert d.abstain and d.reason == "no_candidates"


def test_as_of_before_any_effective_window_abstains(groups):
    # atm_fee windows start 2023 — a 2019 query is in force for neither.
    d = S.select_source(groups["atm_fee"], as_of="2019-01-01")
    assert d.abstain and d.reason == "none_in_effect"


# --------------------------------------------------------------------------- #
# training-export partitioning
# --------------------------------------------------------------------------- #
def test_partition_keeps_stale_and_revoked_out_of_active_export(records):
    part = S.partition_for_export(records)
    active_ids = {S._qid(r) for r in part.active_export}
    for r in part.active_export:
        assert r.get("source_status") not in ("stale", "revoked")
    # Latest mutable + stable + unanswerable behaviour are exported.
    assert {"vf002", "vf006", "vf008", "vf010"} <= active_ids
    # Stale/revoked/superseded go to the versioned archive.
    arch_ids = {S._qid(r) for r in part.versioned_archive}
    assert {"vf001", "vf003", "vf007", "vf009"} <= arch_ids
    # Unresolved conflict is held for review, never trained on.
    held_ids = {S._qid(r) for r in part.held_for_review}
    assert {"vf004", "vf005"} <= held_ids
    # Partition is a true partition of the input.
    assert len(part.active_export) + len(part.versioned_archive) + len(part.held_for_review) == len(records)


def test_partition_holds_rejected_and_pending_rows():
    recs = [
        {"qa_id": "r1", "question": "q", "answer": "a", "answerable": True,
         "fact_volatility": "stable", "source_status": "active", "review_status": "rejected",
         "fact_key": "f1"},
        {"qa_id": "r2", "question": "q2", "answer": "a2", "answerable": True,
         "fact_volatility": "stable", "source_status": "active",
         "review_status": "owner_review_pending", "fact_key": "f2"},
    ]
    part = S.partition_for_export(recs)
    assert not part.active_export
    assert {S._qid(r) for r in part.held_for_review} == {"r1", "r2"}


# --------------------------------------------------------------------------- #
# version-change lineage
# --------------------------------------------------------------------------- #
def test_affected_by_version_change_tracks_qa_and_dataset_version(records):
    out = S.affected_by_version_change(records, new_version="v2", document_id="policy_rate")
    assert out["affected_qa_ids"] == ["vf001"]  # cites policy_rate v1
    assert out["n_affected"] == 1
    assert out["dataset_version"].startswith("ds-")
    # Deterministic.
    again = S.affected_by_version_change(records, new_version="v2", document_id="policy_rate")
    assert again["dataset_version"] == out["dataset_version"]


def test_affected_by_version_change_requires_document_ref(records):
    with pytest.raises(ValueError):
        S.affected_by_version_change(records, new_version="v2")


# --------------------------------------------------------------------------- #
# separate mutable-fact category report
# --------------------------------------------------------------------------- #
def test_mutable_fact_report_is_a_separate_category(records):
    rep = S.mutable_fact_report(records)
    assert rep["category"] == "mutable_fact_recency"
    assert rep["n_mutable_facts"] == 4         # policy_rate, annual_fee, credit_limit, atm_fee
    assert rep["resolved"] == 2 and rep["abstained"] == 2
    assert rep["recency_rate"] == 1.0 and rep["citation_rate"] == 1.0
    assert rep["abstention_rate"] == 0.5


# --------------------------------------------------------------------------- #
# fixture stays schema-valid
# --------------------------------------------------------------------------- #
def test_versioned_fixture_validates_against_schema(records):
    jsonschema = pytest.importorskip("jsonschema")
    with open(_SCHEMA, encoding="utf-8") as fh:
        schema = json.load(fh)
    for rec in records:
        jsonschema.validate(rec, schema)
