"""P1-8: deterministic error taxonomy for dev-set predictions.

Turning failures back into data is important harness evidence — but only if it is
**leakage-safe**. This module classifies a *dev* prediction's failure modes so the
failure-to-data loop (``workflows/.../failure_to_data.py``) can route corrections.

The taxonomy is the dual of the programmatic-verifier rewards
(``evaluation/rewards.py``): where a reward fails, an error category is emitted.
It covers at least the spec-mandated set: grounding, wrong-version, numeric/unit,
abstention, OCR, citation, schema, policy violation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation import rewards as R

# Canonical categories (spec-mandated minimum + a catch-all).
GROUNDING = "grounding"
WRONG_VERSION = "wrong_version"
NUMERIC_UNIT = "numeric_unit"
ABSTENTION = "abstention"
OCR = "ocr"
CITATION = "citation"
SCHEMA = "schema"
POLICY_VIOLATION = "policy_violation"
OTHER = "other"

ERROR_CATEGORIES = (
    GROUNDING, WRONG_VERSION, NUMERIC_UNIT, ABSTENTION, OCR, CITATION,
    SCHEMA, POLICY_VIOLATION, OTHER,
)

# Characters that signal OCR noise (replacement char, stray control/■ boxes).
_OCR_NOISE = re.compile(r"[\uFFFD\u25A0-\u25FF\u2400-\u243F]|[|]{2,}|_{3,}")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _looks_ocr_noisy(text: str) -> bool:
    if not text:
        return False
    if _OCR_NOISE.search(text):
        return True
    # a high ratio of non-word, non-space, non-CJK symbols suggests OCR garbage
    junk = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,%()-원년월일"
                                       or "\uac00" <= c <= "\ud7a3"))
    return len(text) >= 8 and junk / len(text) > 0.4


def _numbers(text: str) -> List[str]:
    return [m.group(0).replace(",", "") for m in _NUM.finditer(text or "")]


@dataclass
class ErrorReport:
    qa_id: Optional[str]
    categories: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_error(self) -> bool:
        return bool(self.categories)


def classify_error(prediction: Dict[str, Any],
                   gold: Optional[Dict[str, Any]] = None,
                   *,
                   latest_version: Optional[str] = None,
                   documents: Optional[Dict[str, Any]] = None) -> ErrorReport:
    """Classify a single dev prediction's failure modes (deterministic).

    ``prediction`` is a Q&A-with-evidence record. ``gold`` (optional) enables
    numeric-mismatch and abstention-direction checks. ``latest_version`` enables
    the wrong-version check; ``documents`` enables full evidence grounding.
    """
    cats: List[str] = []
    detail: Dict[str, Any] = {}
    answerable = prediction.get("answerable", True)
    abstain_cat = prediction.get("category") in R.ABSTAIN_CATEGORIES

    # schema
    if R.reward_schema_compliance(prediction) == 0.0:
        cats.append(SCHEMA)

    # policy / PII
    if R.reward_pii_nonexposure(prediction) == 0.0:
        cats.append(POLICY_VIOLATION)

    # abstention direction (needs to know whether it *should* have abstained)
    if R.reward_abstention(prediction) == 0.0:
        cats.append(ABSTENTION)

    # citation vs grounding (only for answerable, non-abstain records)
    if answerable and not abstain_cat:
        evidence = prediction.get("evidence") or []
        if not evidence or R.reward_evidence_validity(prediction, documents) == 0.0:
            cats.append(CITATION)
        if documents is not None:
            from evaluation.evidence_verifier import verify_record
            if not verify_record(prediction, documents).ok:
                cats.append(GROUNDING)
        elif evidence and R.reward_numeric_consistency(prediction) == 0.0:
            cats.append(GROUNDING)

    # numeric / unit mismatch vs gold
    if gold is not None and gold.get("answerable", True):
        g_nums, p_nums = _numbers(gold.get("answer", "")), _numbers(prediction.get("answer", ""))
        if g_nums and set(g_nums) != set(p_nums):
            cats.append(NUMERIC_UNIT)
            detail["gold_numbers"] = g_nums
            detail["pred_numbers"] = p_nums

    # wrong document version
    if latest_version is not None and R.reward_version_recency(prediction, latest_version) == 0.0:
        cats.append(WRONG_VERSION)

    # OCR noise in the answer or any cited quote
    blob = prediction.get("answer", "") + " " + " ".join(
        e.get("quote", "") or "" for e in (prediction.get("evidence") or []))
    if _looks_ocr_noisy(blob):
        cats.append(OCR)

    # de-dup, stable order
    seen = set()
    ordered = [c for c in ERROR_CATEGORIES if c in cats and not (c in seen or seen.add(c))]
    return ErrorReport(prediction.get("qa_id"), ordered, detail)


def summarize(reports: List[ErrorReport]) -> Dict[str, Any]:
    from collections import Counter
    counter: Counter = Counter()
    for rep in reports:
        counter.update(rep.categories)
    return {
        "n_reports": len(reports),
        "n_with_error": sum(1 for r in reports if r.has_error),
        "by_category": {c: counter.get(c, 0) for c in ERROR_CATEGORIES if counter.get(c, 0)},
    }
