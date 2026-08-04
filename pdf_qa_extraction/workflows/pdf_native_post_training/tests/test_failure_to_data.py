"""P1-8: tests for the leakage-safe failure-to-data loop + error taxonomy."""

import hashlib
import os
import sys

import pytest

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from evaluation import error_taxonomy as ET  # noqa: E402
from workflows.pdf_native_post_training import failure_to_data as F2D  # noqa: E402
from workflows.pdf_native_post_training import review as RV  # noqa: E402
from pdf_qa.provenance import normalize_text, quote_sha256  # noqa: E402

_TS = "2024-01-01T00:00:00Z"


def _ev(quote, version=None):
    e = {"document_sha256": "a" * 64, "page": 1, "element_id": "p1-b1",
         "quote": quote, "quote_sha256": quote_sha256(normalize_text(quote)), "modality": "text"}
    if version is not None:
        e["document_version"] = version
    return e


def _rec(qa_id="d1", answer="매출은 1,250억 원입니다.", answerable=True,
         category="numeric_exact", quote="당기 매출은 1,250억 원으로 집계", **kw):
    r = {"qa_id": qa_id, "question": "매출?", "answer": answer, "answerable": answerable,
         "category": category, "evidence": [_ev(quote)],
         "generation": {"provider": "p", "model": "m"}, "review_status": "approved"}
    r.update(kw)
    return r


# --------------------------------------------------------------------------- #
# error taxonomy — each spec-mandated category is reachable
# --------------------------------------------------------------------------- #
def test_taxonomy_clean_prediction_has_no_error():
    assert classify(_rec()).categories == []


def classify(pred, gold=None, **kw):
    return ET.classify_error(pred, gold, **kw)


def test_taxonomy_detects_each_category():
    assert ET.SCHEMA in classify({**_rec(), "qa_id": None}).categories  # missing required id
    assert ET.CITATION in classify(_rec(evidence=[])).categories
    assert ET.POLICY_VIOLATION in classify(_rec(answer="주민번호 900101-1234567")).categories
    assert ET.ABSTENTION in classify(
        _rec(answerable=False, category="unanswerable", answer="부채비율은 42% 입니다.", evidence=[])).categories
    assert ET.NUMERIC_UNIT in classify(_rec(answer="매출은 9,999억 원입니다."), _rec()).categories
    assert ET.OCR in classify(_rec(answer="매출은 1,250\uFFFD \u25A0\u25A0 원")).categories
    wv = _rec(source_status="active", evidence=[_ev("당기 매출", version="v1")])
    assert ET.WRONG_VERSION in classify(wv, latest_version="v2").categories


def test_summarize_counts_by_category():
    reports = [classify(_rec(evidence=[])), classify(_rec(answer="주민번호 900101-1234567"))]
    s = ET.summarize(reports)
    assert s["n_reports"] == 2 and s["n_with_error"] == 2
    assert s["by_category"].get(ET.CITATION, 0) >= 1


# --------------------------------------------------------------------------- #
# leakage guard — final IDs must never reach corrections/export/train
# --------------------------------------------------------------------------- #
def test_assert_no_final_leakage_direct_and_derived():
    ok = [{"qa_id": "corr-d1", "derived_from": {"dev_qa_id": "d1"}}]
    F2D.assert_no_final_leakage(ok, final_ids={"f1", "f2"})  # no raise
    # direct final id
    with pytest.raises(F2D.FinalLeakageError):
        F2D.assert_no_final_leakage([{"qa_id": "f1"}], final_ids={"f1"})
    # derived from a final id
    with pytest.raises(F2D.FinalLeakageError):
        F2D.assert_no_final_leakage(
            [{"qa_id": "corr-x", "derived_from": {"dev_qa_id": "f2"}}], final_ids={"f2"})


def test_mining_refuses_final_ids_and_non_dev():
    preds = [_rec(qa_id="f1", answer="x")]
    with pytest.raises(F2D.FinalLeakageError):
        F2D.mine_failures(preds, {}, dev_ids={"d1"}, final_ids={"f1"})
    with pytest.raises(ValueError):
        F2D.mine_failures(preds, {}, dev_ids={"d1"}, final_ids=set())  # f1 not in dev


def test_assemble_dataset_version_blocks_final_leak():
    row = {"qa_id": "corr-d1", "derived_from": {"dev_qa_id": "f9"}}
    with pytest.raises(F2D.FinalLeakageError):
        F2D.assemble_dataset_version([row], base_version="v0", final_ids={"f9"})


# --------------------------------------------------------------------------- #
# lineage — each training row traces to a dev failure + source evidence
# --------------------------------------------------------------------------- #
def test_correction_carries_lineage_and_requires_approval():
    pred = _rec(qa_id="d1", answer="매출은 9,999억 원입니다.")  # wrong number
    gold = _rec(qa_id="d1")
    failures = F2D.mine_failures([pred], {"d1": gold}, dev_ids={"d1"}, final_ids={"f1"})
    assert failures and ET.NUMERIC_UNIT in failures[0].categories

    log = RV.ReviewLog()
    corr = F2D.build_correction(failures[0], "매출은 1,250억 원입니다.", "alice",
                                review_log=log, final_ids={"f1"}, timestamp=_TS)
    # lineage back to the dev failure + its source evidence
    assert corr["derived_from"]["dev_qa_id"] == "d1"
    assert ET.NUMERIC_UNIT in corr["derived_from"]["error_categories"]
    assert corr["derived_from"]["evidence"][0]["document_sha256"] == "a" * 64
    # human-approved (edited) via a real review event
    assert corr["review_status"] == "edited"
    assert corr["review_event_id"]
    assert log.status_of(corr["qa_id"]) == "edited"


def test_build_correction_rejects_final_derivation():
    fail = F2D.Failure(dev_qa_id="f1", categories=["numeric_unit"], prediction=_rec(), gold=None,
                       evidence=[_ev("q")])
    with pytest.raises(F2D.FinalLeakageError):
        F2D.build_correction(fail, "fixed", "alice", final_ids={"f1"})


def test_assemble_dataset_version_is_deterministic_and_tracks_lineage():
    fail = F2D.Failure(dev_qa_id="d1", categories=["numeric_unit"], prediction=_rec(), gold=None,
                       evidence=[_ev("q")])
    corr = F2D.build_correction(fail, "fixed", "alice")
    out1 = F2D.assemble_dataset_version([corr], base_version="v0", final_ids={"f1"})
    out2 = F2D.assemble_dataset_version([corr], base_version="v0", final_ids={"f1"})
    assert out1["dataset_version"] == out2["dataset_version"]
    assert out1["dataset_version"].startswith("ds-")
    assert out1["lineage"][0]["dev_qa_id"] == "d1"


# --------------------------------------------------------------------------- #
# dev-reuse counting (overfitting visibility)
# --------------------------------------------------------------------------- #
def test_dev_reuse_ledger_counts_rounds(tmp_path):
    led = F2D.DevReuseLedger()
    led.record_round(["d1", "d2"])
    led.record_round(["d1"])
    assert led.counts() == {"d1": 2, "d2": 1}
    assert led.max_reuse() == 2
    p = str(tmp_path / "reuse.json")
    led.save(p)
    reloaded = F2D.DevReuseLedger.load(p)
    assert reloaded.report()["rounds"] == 2
    assert reloaded.counts() == {"d1": 2, "d2": 1}


# --------------------------------------------------------------------------- #
# one-time protected final evaluation with access record
# --------------------------------------------------------------------------- #
def test_final_access_ledger_single_scoring(tmp_path):
    p = str(tmp_path / "final_access.jsonl")
    led = F2D.FinalAccessLedger()
    led.access({"f1", "f2"}, "design-time peek forbidden -> none", scoring=False, path=p)
    led.access({"f1", "f2"}, "final scoring after all decisions", scoring=True, path=p)
    led.assert_single_scoring()  # exactly one scoring so far
    assert len(led.records()) == 2
    # a second scoring pass must be blocked
    led.access({"f1", "f2"}, "oops second scoring", scoring=True)
    with pytest.raises(RuntimeError):
        led.assert_single_scoring()
    # persisted access records survive reload
    reloaded = F2D.FinalAccessLedger.load(p)
    assert len(reloaded.records()) == 2
    assert len(reloaded.scorings()) == 1
