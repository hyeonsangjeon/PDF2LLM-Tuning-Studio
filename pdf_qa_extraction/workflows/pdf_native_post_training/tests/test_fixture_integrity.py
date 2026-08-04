import json
import os
import sys

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pdf_qa import pii  # noqa: E402
from pdf_qa.provenance import parse_pdf, quote_sha256  # noqa: E402
from evaluation.evidence_verifier import verify_dataset, verify_record  # noqa: E402

_DEMO = os.path.join(_ROOT, "workflows", "pdf_native_post_training", "public_finance_demo")


def _load_gold():
    with open(os.path.join(_DEMO, "gold_qa.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _docs():
    doc = parse_pdf(os.path.join(_DEMO, "docs", "finance_report_v1.pdf"), version="v1")
    return {doc.sha256: doc}, doc


def test_gold_qa_has_full_evidence_integrity():
    gold = _load_gold()
    docs, _ = _docs()
    rep = verify_dataset(gold, docs)
    assert rep["evidence_address_integrity"] == 1.0, rep["failures"]
    assert rep["total"] >= 24


def test_all_embedded_canaries_are_mechanically_fake():
    ledger = json.load(open(os.path.join(_DEMO, "canary_ledger.json")))
    for kind, value in ledger["canaries"].items():
        assert pii.is_mechanically_fake(kind, value), (kind, value)
    # and the corpus as a whole has no *real* PII
    doc = parse_pdf(os.path.join(_DEMO, "docs", "finance_report_v1.pdf"))
    text = " ".join(e.text for e in doc.elements)
    assert pii.has_real_pii(text) is False


def test_tampered_quote_fails_verification():
    gold = _load_gold()
    docs, _ = _docs()
    rec = next(dict(r) for r in gold if r.get("evidence"))
    rec["evidence"] = [dict(rec["evidence"][0])]
    rec["evidence"][0]["quote"] = "완전히 조작된 인용문입니다"  # not in the document
    rec["evidence"][0]["quote_sha256"] = quote_sha256(rec["evidence"][0]["quote"])
    assert verify_record(rec, docs).ok is False


def test_fake_element_id_fails_verification():
    gold = _load_gold()
    docs, _ = _docs()
    rec = next(dict(r) for r in gold if r.get("evidence"))
    rec["evidence"] = [dict(rec["evidence"][0])]
    rec["evidence"][0]["element_id"] = "p9-b99"  # never produced by the parser
    assert verify_record(rec, docs).ok is False


def test_checksums_match_shipped_files():
    checks = {}
    with open(os.path.join(_DEMO, "checksums.sha256"), encoding="utf-8") as fh:
        for line in fh:
            digest, rel = line.strip().split("  ", 1)
            checks[rel] = digest
    from pdf_qa.run_bundle import sha256_file
    for rel, digest in checks.items():
        assert sha256_file(os.path.join(_DEMO, rel)) == digest, rel
