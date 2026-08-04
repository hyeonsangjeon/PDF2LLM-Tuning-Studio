"""Unit tests for the PDF-native metric contract (spec P1-5).

Pure CPU, no network/GPU. Verifies every metric in the contract plus the
leakage-audited split enforcement.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evaluation import pdf_native as P  # noqa: E402
from evaluation.pdf_native import SplitLeakageError  # noqa: E402

_CORPUS = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "workflows", "pdf_native_post_training", "public_finance_demo"))


# --------------------------------------------------------------------------- text
def test_f1_and_em():
    assert P.exact_match("서울입니다", "서울입니다") == 1.0
    assert P.exact_match("서울", "부산") == 0.0
    assert P.f1_score("서울특별시", "서울특별시") == 1.0
    assert P.f1_score("완전히 다른 답", "서울") == 0.0
    partial = P.f1_score("서울특별시", "서울")
    assert 0.0 < partial < 1.0


def test_normalize_strips_punct_and_case():
    assert P.normalize_answer("  Hello, World!! ") == "hello world"


# --------------------------------------------------------------------------- typed
def test_numeric_exact():
    assert P.numeric_exact("정답은 1,250억 원", "1250") is True
    assert P.numeric_exact("정답은 300", "1250") is False
    assert P.numeric_exact("숫자 없는 답", "숫자 없는 골드") is None  # gold has no number


def test_unit_exact():
    assert P.unit_exact("연 3.5%", "3.5%") is True
    assert P.unit_exact("3.5 원", "3.5%") is False
    assert P.unit_exact("서울", "부산") is None


def test_date_exact():
    assert P.date_exact("기준일은 2024년", "2024년") is True
    assert P.date_exact("2023년", "2024년") is False
    assert P.date_exact("숫자없음", "날짜없는골드") is None


# --------------------------------------------------------------------------- citation
def test_citation_page():
    gold = [{"page": 2}, {"page": 3}]
    assert P.citation_page_correct([{"page": 2}], gold) is True
    assert P.citation_page_correct([{"page": 9}], gold) is False
    assert P.citation_page_correct([], gold) is False
    assert P.citation_page_correct([{"page": 2}], []) is None  # no gold pages


def test_citation_span_by_id_hash_and_substring():
    gold = [{"element_id": "e12", "quote": "연간 매출액은 1,250억 원", "quote_sha256": "abc"}]
    assert P.citation_span_correct([{"element_id": "e12"}], gold) is True
    assert P.citation_span_correct([{"quote_sha256": "abc"}], gold) is True
    assert P.citation_span_correct([{"quote": "매출액은 1,250억"}], gold) is True  # substring
    assert P.citation_span_correct([{"quote": "전혀 다른 텍스트"}], gold) is False
    assert P.citation_span_correct([], gold) is False
    assert P.citation_span_correct([{"element_id": "e12"}], []) is None


# --------------------------------------------------------------------------- retrieval
def test_lexical_retrieve_deterministic_and_recall():
    corpus = [
        {"element_id": "e1", "text": "연간 매출액은 1250억 원 입니다"},
        {"element_id": "e2", "text": "영업이익률은 12 퍼센트"},
        {"element_id": "e3", "text": "직원 수는 300명"},
    ]
    top = P.lexical_retrieve(corpus, "연간 매출액은 얼마인가", k=2)
    assert top[0] == "e1"
    # deterministic: identical call -> identical order
    assert P.lexical_retrieve(corpus, "연간 매출액은 얼마인가", k=2) == top
    assert P.recall_at_k(["e1"], top, 2) == 1.0
    assert P.recall_at_k(["e9"], top, 2) == 0.0
    assert P.recall_at_k([], top, 2) is None  # no gold ids -> undefined


# --------------------------------------------------------------------------- abstain/schema/ground/pii
def test_predicted_abstained():
    assert P.predicted_abstained({"abstained": True}) is True
    assert P.predicted_abstained({"answer": "확인할 수 없습니다"}) is True
    assert P.predicted_abstained({"answer": "서울입니다"}) is False


def test_schema_valid():
    assert P.schema_valid({"qa_id": "q1", "answer": "x"}) is True
    assert P.schema_valid({"qa_id": "q1", "answer": ""}) is False
    assert P.schema_valid({"qa_id": "q1", "answer": "x", "citations": "notalist"}) is False
    assert P.schema_valid("notadict") is False


def test_is_grounded():
    ev = [{"quote": "연간 매출액은 1,250억 원"}]
    assert P.is_grounded({"answer": "1,250억 원"}, ev) is True   # numeric subset
    assert P.is_grounded({"answer": "9,999억 원"}, ev) is False  # numeric not in evidence
    assert P.is_grounded({"answer": "모르겠습니다"}, ev) is True  # abstain always grounded


def test_pii_leaked():
    assert P.pii_leaked({"answer": "연락처 canary@example.com"}, ["canary@example.com"]) is True
    assert P.pii_leaked({"answer": "일반 답변"}, ["canary@example.com"]) is False


# --------------------------------------------------------------------------- scoring
def test_score_example_unanswerable_nulls_qa_metrics():
    gold = {"qa_id": "u1", "answerable": False, "answer": "", "category": "unanswerable"}
    rec = P.score_example(gold, {"qa_id": "u1", "answer": "확인할 수 없습니다"})
    assert rec["em"] is None and rec["f1"] is None
    assert rec["abstained"] is True
    assert rec["error_categories"] == []  # correct refusal -> no error


def test_failure_categories():
    # perfect answerable -> no error
    good = P.score_example({"qa_id": "a", "answerable": True, "answer": "서울입니다",
                            "evidence": [{"quote": "서울입니다", "page": 1}]},
                           {"qa_id": "a", "answer": "서울입니다",
                            "citations": [{"page": 1, "quote": "서울입니다"}]})
    assert good["error_categories"] == []
    # wrong answerable -> wrong_answer + citation + grounding
    bad = P.score_example({"qa_id": "b", "answerable": True, "answer": "서울입니다",
                           "evidence": [{"quote": "서울입니다", "page": 1}]},
                          {"qa_id": "b", "answer": "완전히 틀린 답", "citations": []})
    assert "wrong_answer" in bad["error_categories"]
    assert "citation" in bad["error_categories"]
    # answered an unanswerable -> missed_abstention
    missed = P.score_example({"qa_id": "u", "answerable": False, "answer": ""},
                             {"qa_id": "u", "answer": "아무 답변"})
    assert "missed_abstention" in missed["error_categories"]
    # abstained on an answerable -> over_abstention
    over = P.score_example({"qa_id": "o", "answerable": True, "answer": "서울",
                            "evidence": [{"quote": "서울", "page": 1}]},
                           {"qa_id": "o", "answer": "확인할 수 없습니다"})
    assert "over_abstention" in over["error_categories"]


def test_abstention_precision_recall():
    recs = [
        {"abstained": True, "answerable": False},   # tp
        {"abstained": True, "answerable": True},    # fp
        {"abstained": False, "answerable": False},  # fn
        {"abstained": False, "answerable": True},   # tn
    ]
    pr = P.abstention_precision_recall(recs)
    assert pr["tp"] == 1 and pr["fp"] == 1 and pr["fn"] == 1
    assert pr["precision"] == 0.5 and pr["recall"] == 0.5


def test_aggregate_auto_from_raw():
    recs = [
        P.score_example({"qa_id": "1", "answerable": True, "answer": "서울",
                         "category": "single_fact", "evidence": [{"quote": "서울", "page": 1}]},
                        {"qa_id": "1", "answer": "서울", "citations": [{"page": 1, "quote": "서울"}]}),
        P.score_example({"qa_id": "2", "answerable": False, "answer": "",
                         "category": "unanswerable"},
                        {"qa_id": "2", "answer": "확인할 수 없습니다"}),
    ]
    agg = P.aggregate(recs)
    assert agg["n_examples"] == 2 and agg["n_answerable"] == 1 and agg["n_unanswerable"] == 1
    assert agg["em"] == 1.0 and agg["f1"] == 1.0
    assert set(agg["per_category"]) == {"single_fact", "unanswerable"}
    assert agg["abstention"]["recall"] == 1.0


# --------------------------------------------------------------------------- dataset on real fixture
def _load_gold():
    import json
    with open(os.path.join(_CORPUS, "gold_qa.jsonl"), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_score_dataset_perfect_vs_wrong_on_fixture():
    gold = _load_gold()
    perfect = {g["qa_id"]: {"qa_id": g["qa_id"], "answer": g["answer"],
                            "citations": [{"page": e.get("page"), "element_id": e.get("element_id"),
                                           "quote": e.get("quote"), "quote_sha256": e.get("quote_sha256")}
                                          for e in g.get("evidence", [])]} for g in gold}
    out = P.score_dataset(gold, perfect, pii_terms=["canary@example.com"])
    a = out["aggregate"]
    assert a["em"] == 1.0 and a["f1"] == 1.0
    assert a["citation_page_accuracy"] == 1.0
    assert a["pii_leakage_rate"] == 0.0
    assert a["abstention"]["recall"] == 1.0
    assert len(out["per_example"]) == len(gold)

    wrong = {g["qa_id"]: {"qa_id": g["qa_id"], "answer": "오답", "citations": []} for g in gold}
    aw = P.score_dataset(gold, wrong)["aggregate"]
    assert aw["f1"] < a["f1"]
    assert aw["failure_taxonomy"]["n_with_error"] > 0
    assert "wrong_answer" in aw["failure_taxonomy"]["by_category"]


def test_score_dataset_retrieval_populates_recall():
    gold = _load_gold()
    corpus = []
    seen = set()
    for g in gold:
        for e in g.get("evidence", []):
            eid = e.get("element_id")
            if eid and eid not in seen:
                seen.add(eid)
                corpus.append({"element_id": eid, "text": e.get("quote", "")})
    out = P.score_dataset(gold, {}, corpus=corpus, k=5)
    a = out["aggregate"]
    assert "retrieval_recall_at_k" in a
    assert a["retrieval_k"] == 5


# --------------------------------------------------------------------------- leakage audit
def _rows(family, split, ids, version="v1", ev=None):
    return [{"qa_id": i, "document_family_id": family, "split": split,
             "document_version": version, "evidence": ev or []} for i in ids]


def test_leakage_audit_passes_when_disjoint():
    splits = {"dev": _rows("famA", "dev", ["a1", "a2"]),
              "regression": _rows("famB", "regression", ["b1"])}
    audit = P.assert_no_split_leakage(splits)
    assert audit["disjoint"] is True and audit["intersection_size"] == 0
    assert audit["n_families"] == 2


def test_leakage_audit_raises_on_family_overlap():
    splits = {"dev": _rows("famA", "dev", ["a1"]),
              "final": _rows("famA", "final", ["a2"])}  # same family both splits
    with pytest.raises(SplitLeakageError):
        P.assert_no_split_leakage(splits)


def test_leakage_audit_raises_on_span_overlap():
    ev = [{"quote_sha256": "SHARED"}]
    splits = {"dev": _rows("famA", "dev", ["a1"], ev=ev),
              "regression": _rows("famB", "regression", ["b1"], ev=ev)}  # same source span
    with pytest.raises(SplitLeakageError):
        P.assert_no_split_leakage(splits)


def test_leakage_audit_keeps_versions_together():
    splits = {"dev": (_rows("famA", "dev", ["a1"], version="v1") +
                      _rows("famA", "dev", ["a2"], version="v2"))}
    audit = P.assert_no_split_leakage(splits)
    assert audit["family_versions_together"]["famA"] == ["v1", "v2"]
