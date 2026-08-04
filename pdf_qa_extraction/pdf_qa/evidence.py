"""Build evidence records and extract groundable tokens (numbers/dates/currency).

Kept dependency-free so both the generation pipeline and the evaluator can use
it. ``build_evidence`` never invents an element id: it must be given one the
parser produced (see :mod:`pdf_qa.provenance`).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .provenance import Document, Element, normalize_text, quote_sha256

_NUM_RE = re.compile(r"\d[\d,\.]*")
_DATE_RE = re.compile(r"\d{4}\s*년|\d{1,2}\s*월|\d{1,2}\s*일|\d{4}-\d{2}-\d{2}")
_CURRENCY_RE = re.compile(r"(억\s*달러|억\s*불|조\s*원|억\s*원|만\s*원|달러|원|USD|KRW|%|퍼센트)")


def normalize_number(tok: str) -> str:
    return tok.replace(",", "").rstrip(".")


def extract_numbers(text: str) -> List[str]:
    return [normalize_number(m.group(0)) for m in _NUM_RE.finditer(text or "")]


def extract_dates(text: str) -> List[str]:
    return [normalize_text(m.group(0)) for m in _DATE_RE.finditer(text or "")]


def groundable_tokens(text: str) -> List[str]:
    """Numbers + dates that a factual answer should be traceable to."""
    return extract_numbers(text) + extract_dates(text)


def build_evidence(
    element: Element,
    quote: str,
    document_sha256: str,
    document_version: Optional[str] = None,
) -> Dict:
    """Construct a schema-valid evidence item anchored to a real element."""
    nq = normalize_text(quote)
    if nq not in element.text:
        raise ValueError(f"quote not found verbatim in element {element.element_id}")
    return {
        "document_sha256": document_sha256,
        "document_version": document_version,
        "page": element.page,
        "element_id": element.element_id,
        "bbox": [round(float(x), 2) for x in element.bbox],
        "quote": nq,
        "quote_sha256": quote_sha256(nq),
        "modality": element.modality,
        "section_path": element.section_path,
        "chunk_id": element.chunk_id,
    }


def make_qa(
    qa_id: str,
    question: str,
    answer: str,
    evidence: List[Dict],
    provider: str,
    model: str,
    *,
    category: str = "single_fact",
    answerable: bool = True,
    generation_mode: str = "recorded_replay",
    model_revision: Optional[str] = None,
    prompt_sha256: Optional[str] = None,
    fact_volatility: str = "stable",
    document_version: Optional[str] = None,
    source_status: str = "active",
    review_status: str = "owner_review_pending",
) -> Dict:
    return {
        "qa_id": qa_id,
        "question": question,
        "answer": answer,
        "answerable": answerable,
        "category": category,
        "fact_volatility": fact_volatility,
        "document_version": document_version,
        "source_status": source_status,
        "evidence": evidence,
        "generation": {
            "provider": provider,
            "model": model,
            "model_revision": model_revision,
            "prompt_sha256": prompt_sha256,
            "generation_mode": generation_mode,
        },
        "review_status": review_status,
    }


def index_documents(docs: List[Document]) -> Dict[str, Document]:
    return {d.sha256: d for d in docs}
