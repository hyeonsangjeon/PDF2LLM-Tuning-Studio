"""Tests for the deterministic evidence-address verifier (P0-8)."""
from __future__ import annotations

import os
import sys

REPO_PDFQA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_PDFQA)

from evaluation.evidence_verifier import verify_dataset, verify_record  # noqa: E402
from pdf_qa.evidence import build_evidence, make_qa  # noqa: E402
from pdf_qa.provenance import Document, Element  # noqa: E402


def _doc():
    els = [
        Element("p1-b0", 1, (10, 10, 500, 30), "외환보유액이 4,000억불을 상회하고 순대외금융자산이 1조불에 이른다", "text"),
        Element("p2-b0", 2, (10, 10, 500, 30), "선물환포지션 한도를 50%에서 75%로 상향 조정한다", "text"),
    ]
    return Document(path="x.pdf", sha256="doc1", version="v1", n_pages=2, elements=els)


def _index(doc):
    return {doc.sha256: doc}


def test_valid_record_passes():
    doc = _doc()
    el = doc.elements[0]
    ev = build_evidence(el, "외환보유액이 4,000억불을 상회", "doc1", "v1")
    qa = make_qa("qa1", "외환보유액은?", "외환보유액이 4,000억불을 상회한다.", [ev], "ollama", "m", category="numeric_exact")
    r = verify_record(qa, _index(doc))
    assert r.ok, r.reasons


def test_fake_element_id_rejected():
    doc = _doc()
    el = doc.elements[0]
    ev = build_evidence(el, "외환보유액이 4,000억불을 상회", "doc1", "v1")
    ev["element_id"] = "p9-b99"  # invented
    qa = make_qa("qa2", "q", "a 4000", [ev], "ollama", "m")
    r = verify_record(qa, _index(doc))
    assert not r.ok and r.checks.get("element_exists") is False


def test_tampered_quote_rejected():
    doc = _doc()
    el = doc.elements[0]
    ev = build_evidence(el, "외환보유액이 4,000억불을 상회", "doc1", "v1")
    ev["quote"] = "외환보유액이 9,999억불을 상회"  # not in element; hash now stale too
    qa = make_qa("qa3", "q", "a", [ev], "ollama", "m")
    r = verify_record(qa, _index(doc))
    assert not r.ok
    assert r.checks.get("quote_present") is False


def test_wrong_page_rejected():
    doc = _doc()
    el = doc.elements[0]
    ev = build_evidence(el, "외환보유액이 4,000억불을 상회", "doc1", "v1")
    ev["page"] = 2
    qa = make_qa("qa4", "q", "a 4000", [ev], "ollama", "m")
    r = verify_record(qa, _index(doc))
    assert not r.ok and r.checks.get("page_match") is False


def test_numeric_hallucination_rejected():
    doc = _doc()
    el = doc.elements[1]
    ev = build_evidence(el, "선물환포지션 한도를 50%에서 75%로", "doc1", "v1")
    # answer invents 999 which is not in evidence
    qa = make_qa("qa5", "q", "한도를 999%로 올린다", [ev], "ollama", "m", category="numeric_exact")
    r = verify_record(qa, _index(doc))
    assert not r.ok and r.checks.get("numeric_grounded") is False


def test_unanswerable_needs_no_evidence():
    doc = _doc()
    qa = make_qa("qa6", "2027년 금리는?", "문서에 근거가 없어 답할 수 없습니다.", [], "ollama", "m",
                 category="unanswerable", answerable=False)
    r = verify_record(qa, _index(doc))
    assert r.ok, r.reasons


def test_dataset_integrity_metric():
    doc = _doc()
    el = doc.elements[0]
    good = make_qa("g", "q", "외환보유액이 4,000억불을 상회한다.",
                   [build_evidence(el, "외환보유액이 4,000억불을 상회", "doc1", "v1")], "ollama", "m",
                   category="numeric_exact")
    bad_ev = build_evidence(el, "외환보유액이 4,000억불을 상회", "doc1", "v1")
    bad_ev["element_id"] = "p9-bX"
    bad = make_qa("b", "q", "a", [bad_ev], "ollama", "m")
    report = verify_dataset([good, bad], _index(doc))
    assert report["total"] == 2 and report["passed"] == 1
    assert report["evidence_address_integrity"] == 0.5
