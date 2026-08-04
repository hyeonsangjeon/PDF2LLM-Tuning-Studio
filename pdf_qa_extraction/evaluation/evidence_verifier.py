"""Deterministic evidence-address verifier (P0-8).

Given Q&A-with-evidence records and the parsed source documents, this checks the
*mechanical* integrity of every citation:

* the referenced ``element_id`` was actually produced by the parser,
* the cited ``page`` matches that element's page,
* the ``quote`` appears verbatim in that element (after NFKC/space normalisation),
* ``quote_sha256`` matches the quote,
* for factual (answerable, non-calculation) categories, the numbers/dates in the
  answer are present in the union of the cited quotes (numeric grounding).

It reports ``evidence_address_integrity`` = passed / total. That is an address /
hash integrity metric, NOT a claim of semantic groundedness or answer
correctness (see the schema notes and P0-8 boundary).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from pdf_qa.evidence import groundable_tokens
from pdf_qa.provenance import Document, normalize_text, quote_sha256

# categories where answer numbers need not be verbatim in the evidence
_NON_GROUNDED_CATEGORIES = {"unanswerable", "calculation", "prompt_injection", "conflicting_evidence"}


@dataclass
class RecordResult:
    qa_id: str
    ok: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


def verify_record(qa: Dict, documents: Dict[str, Document]) -> RecordResult:
    res = RecordResult(qa_id=qa.get("qa_id", "?"), ok=True, checks={}, reasons=[])
    evidence = qa.get("evidence") or []

    if not evidence and qa.get("answerable", True) and qa.get("category") != "unanswerable":
        res.ok = False
        res.reasons.append("answerable record has no evidence")
        res.checks["has_evidence"] = False
        return res

    all_quotes = []
    for i, ev in enumerate(evidence):
        doc = documents.get(ev.get("document_sha256"))
        prefix = f"evidence[{i}]"
        if doc is None:
            res.ok = False
            res.reasons.append(f"{prefix}: unknown document_sha256")
            continue
        el = doc.by_id().get(ev.get("element_id"))
        if el is None:
            res.ok = False
            res.checks["element_exists"] = False
            res.reasons.append(f"{prefix}: element_id {ev.get('element_id')!r} not produced by parser")
            continue
        res.checks["element_exists"] = res.checks.get("element_exists", True)

        if el.page != ev.get("page"):
            res.ok = False
            res.checks["page_match"] = False
            res.reasons.append(f"{prefix}: page {ev.get('page')} != element page {el.page}")
        else:
            res.checks.setdefault("page_match", True)

        quote = normalize_text(ev.get("quote", ""))
        if quote not in el.text:
            res.ok = False
            res.checks["quote_present"] = False
            res.reasons.append(f"{prefix}: quote not found verbatim in element")
        else:
            res.checks.setdefault("quote_present", True)
            all_quotes.append(quote)

        if quote_sha256(quote) != ev.get("quote_sha256"):
            res.ok = False
            res.checks["quote_hash"] = False
            res.reasons.append(f"{prefix}: quote_sha256 mismatch")
        else:
            res.checks.setdefault("quote_hash", True)

    # numeric grounding for factual answers
    if qa.get("answerable", True) and qa.get("category") not in _NON_GROUNDED_CATEGORIES:
        ev_tokens = set()
        for q in all_quotes:
            ev_tokens.update(groundable_tokens(q))
        missing = [t for t in groundable_tokens(qa.get("answer", "")) if t not in ev_tokens]
        if missing:
            res.ok = False
            res.checks["numeric_grounded"] = False
            res.reasons.append(f"answer tokens not in evidence: {missing}")
        else:
            res.checks["numeric_grounded"] = True

    return res


def verify_dataset(qas: List[Dict], documents: Dict[str, Document]) -> Dict:
    results = [verify_record(q, documents) for q in qas]
    passed = [r for r in results if r.ok]
    total = len(results)
    return {
        "total": total,
        "passed": len(passed),
        "failed": total - len(passed),
        "evidence_address_integrity": (len(passed) / total) if total else 1.0,
        "failures": [
            {"qa_id": r.qa_id, "reasons": r.reasons} for r in results if not r.ok
        ],
    }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps({"usage": "import evaluation.evidence_verifier"}, indent=2))
    sys.exit(0)
