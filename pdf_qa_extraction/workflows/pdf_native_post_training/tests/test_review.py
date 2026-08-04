"""P1-7: tests for the local review workflow (approval queue + export)."""

import json
import os
import sys

import pytest

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from workflows.pdf_native_post_training import review as RV  # noqa: E402
from workflows.pdf_native_post_training import source_selection as S  # noqa: E402

_DEMO = os.path.join(os.path.dirname(__file__), "..", "public_finance_demo")
_GOLD = os.path.join(_DEMO, "gold_qa.jsonl")
_VERSIONED = os.path.join(_DEMO, "versioned_facts.jsonl")
_SCHEMA = os.path.join(_PKG, "workflows", "pdf_native_post_training", "schemas",
                       "review_event.schema.json")

_TS = "2024-01-01T00:00:00Z"


@pytest.fixture
def gold():
    return S.load_records(_GOLD)[:5]


# --------------------------------------------------------------------------- #
# approval is a projection over events, not the JSONL field
# --------------------------------------------------------------------------- #
def test_status_projects_from_events_ignoring_jsonl_field(gold):
    gold[0]["review_status"] = "approved"          # lie in the file
    log = RV.ReviewLog()
    # no event for gold[0] -> must be unreviewed
    assert log.status_of(gold[0]["qa_id"]) == "owner_review_pending"
    RV.accept(log, gold[1], "alice", timestamp=_TS)
    assert log.status_of(gold[1]["qa_id"]) == "approved"


def test_unreviewed_and_rejected_never_reach_training_export(gold):
    log = RV.ReviewLog()
    RV.accept(log, gold[0], "alice", timestamp=_TS)
    RV.reject(log, gold[1], "bob", "ungrounded", timestamp=_TS)
    # gold[2..4] unreviewed
    exp = RV.export_training(gold, log)
    train_ids = {r["qa_id"] for r in exp.train}
    assert train_ids == {gold[0]["qa_id"]}
    # completion condition: zero rejected/unreviewed rows in the train export
    assert all(r["review_status"] in ("approved", "edited") for r in exp.train)
    assert {r["qa_id"] for r in exp.quarantine} == {gold[1]["qa_id"]}
    assert {r["qa_id"] for r in exp.pending} == {g["qa_id"] for g in gold[2:]}


# --------------------------------------------------------------------------- #
# edits preserve the original value + diff
# --------------------------------------------------------------------------- #
def test_edit_preserves_original_and_diff(gold):
    log = RV.ReviewLog()
    orig = gold[0]["answer"]
    RV.edit(log, gold[0], "alice", {"answer": "정정된 답변입니다."}, timestamp=_TS)
    exp = RV.export_training(gold, log)
    row = next(r for r in exp.train if r["qa_id"] == gold[0]["qa_id"])
    assert row["answer"] == "정정된 답변입니다."
    assert row["_review_original"]["answer"] == orig
    assert row["_review_edits"][0]["old"] == orig
    assert row["review_status"] == "edited"


# --------------------------------------------------------------------------- #
# traceability: approved row -> reviewer event + source evidence
# --------------------------------------------------------------------------- #
def test_every_approved_row_is_traceable(gold):
    log = RV.ReviewLog()
    RV.accept(log, gold[0], "alice", timestamp=_TS)
    exp = RV.export_training(gold, log)
    for row in exp.train:
        assert row["review_event_id"]
        assert row["reviewer"] == "alice"
        t = RV.trace(row["qa_id"], log, gold)
        assert t["approved"] and t["traceable"]
        assert t["events"] and t["evidence"]


# --------------------------------------------------------------------------- #
# source-snippet redaction for readers without source access
# --------------------------------------------------------------------------- #
def test_report_redacts_source_snippet_without_access(gold):
    red = RV.redact_for_report(gold[0], has_source_access=False)
    assert red["_source_redacted"] is True
    assert all(e.get("quote") is None for e in red["evidence"])
    # addresses (hashes/page) are kept for auditing
    assert all(e.get("document_sha256") for e in red["evidence"])
    full = RV.redact_for_report(gold[0], has_source_access=True)
    assert any(e.get("quote") for e in full["evidence"])


# --------------------------------------------------------------------------- #
# append-only integrity + chaining
# --------------------------------------------------------------------------- #
def test_event_chain_across_multiple_actions(gold):
    log = RV.ReviewLog()
    qa = gold[0]["qa_id"]
    RV.accept(log, gold[0], "alice", timestamp=_TS)
    RV.reopen(log, gold[0], "bob", timestamp=_TS)
    RV.reject(log, gold[0], "carol", "wrong_version", timestamp=_TS)
    evs = log.events_for(qa)
    assert [e["action"] for e in evs] == ["accept", "reopen", "reject"]
    assert evs[0]["prev_event_id"] is None
    assert evs[1]["prev_event_id"] == evs[0]["event_id"]
    assert evs[2]["prev_event_id"] == evs[1]["event_id"]
    assert log.status_of(qa) == "rejected"
    log.verify()


def test_verify_detects_tampered_event(gold):
    log = RV.ReviewLog()
    RV.accept(log, gold[0], "alice", timestamp=_TS)
    tampered = dict(log.events()[0])
    tampered["reviewer"] = "mallory"       # content changed, hash not updated
    bad = RV.ReviewLog([tampered])
    with pytest.raises(ValueError):
        bad.verify()


def test_verify_detects_reordered_chain(gold):
    log = RV.ReviewLog()
    RV.accept(log, gold[0], "alice", timestamp=_TS)
    RV.reopen(log, gold[0], "bob", timestamp=_TS)
    reordered = RV.ReviewLog(list(reversed(log.events())))
    with pytest.raises(ValueError):
        reordered.verify()


def test_reject_requires_reason(gold):
    with pytest.raises(ValueError):
        RV.make_event(gold[0]["qa_id"], "reject", "alice")
    with pytest.raises(ValueError):
        RV.make_event(gold[0]["qa_id"], "accept", "alice", reject_reason="not_a_reason")


# --------------------------------------------------------------------------- #
# composition with P1-6 source selection
# --------------------------------------------------------------------------- #
def test_approved_but_stale_source_excluded_by_source_selection():
    recs = S.load_records(_VERSIONED)
    vf001 = next(r for r in recs if r["qa_id"] == "vf001")  # stale, superseded
    log = RV.ReviewLog()
    RV.accept(log, vf001, "alice", timestamp=_TS)
    # even though a human approved it, the stale/superseded source is excluded
    exp = RV.export_training(recs, log, apply_source_selection=True)
    assert "vf001" not in {r["qa_id"] for r in exp.train}
    # disabling source-selection lets the human approval through
    exp2 = RV.export_training(recs, log, apply_source_selection=False)
    assert "vf001" in {r["qa_id"] for r in exp2.train}


# --------------------------------------------------------------------------- #
# events validate against the schema
# --------------------------------------------------------------------------- #
def test_events_validate_against_schema(gold):
    jsonschema = pytest.importorskip("jsonschema")
    with open(_SCHEMA, encoding="utf-8") as fh:
        schema = json.load(fh)
    log = RV.ReviewLog()
    RV.accept(log, gold[0], "alice", timestamp=_TS)
    RV.edit(log, gold[1], "bob", {"answer": "x"}, timestamp=_TS)
    RV.reject(log, gold[2], "carol", "pii_exposure", timestamp=_TS)
    for e in log.events():
        jsonschema.validate(e, schema)


# --------------------------------------------------------------------------- #
# file round-trip + CLI export
# --------------------------------------------------------------------------- #
def test_log_file_roundtrip_and_verify(tmp_path, gold):
    path = str(tmp_path / "review.jsonl")
    log = RV.ReviewLog()
    log.append_to_file(RV.make_event(gold[0]["qa_id"], "accept", "alice", timestamp=_TS), path)
    reloaded = RV.ReviewLog.load(path)      # load() calls verify()
    assert reloaded.status_of(gold[0]["qa_id"]) == "approved"


def test_cli_accept_then_export(tmp_path, gold):
    recs_path = str(tmp_path / "gen.jsonl")
    with open(recs_path, "w", encoding="utf-8") as fh:
        for r in gold:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log_path = str(tmp_path / "review.jsonl")
    out_dir = str(tmp_path / "out")
    assert RV.main(["--log", log_path, "accept", "--records", recs_path,
                    "--qa-id", gold[0]["qa_id"], "--reviewer", "alice"]) == 0
    assert RV.main(["--log", log_path, "export", "--records", recs_path,
                    "--out", out_dir]) == 0
    train = [json.loads(l) for l in open(os.path.join(out_dir, "train_approved.jsonl"))]
    assert {r["qa_id"] for r in train} == {gold[0]["qa_id"]}
    # verify CLI passes on the produced log
    assert RV.main(["--log", log_path, "verify"]) == 0
