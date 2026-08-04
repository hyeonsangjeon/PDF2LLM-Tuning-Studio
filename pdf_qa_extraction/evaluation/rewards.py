"""Programmatic-verifier reward components (P2-2 — design first; RL is *planned*).

This module defines **deterministic**, unit-tested reward functions and a
``RewardCard`` registry for a *possible future* reinforcement-learning stage.
It exists so the reward interface and its failure modes are pinned down **before**
any GRPO/PPO is added — see ``docs/RL_EXPERIMENT_PLAN.md``.

Nothing here trains or runs RL, and no RL result is implied. The status is
``planned`` and the intended method, if the feasibility gates are ever met, is a
**programmatic-verifier RL** (rule/verifier rewards) — *not* RLHF/PPO/DPO on
human-preference data.

Each ``reward_*`` maps a Q&A-with-evidence record (see
``pdf_qa/schemas/qa_with_evidence.schema.json``) to a score in ``[0, 1]`` (higher
is better). ``length_penalty`` returns a penalty in ``[0, 1]`` (a length-bias /
reward-hacking guard). ``score_record`` aggregates them with explicit weights and
applies the penalty multiplicatively. All rewards are pure functions of the
record (+ optional context) and use only public/synthetic inputs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pdf_qa.evidence import extract_numbers, groundable_tokens
from pdf_qa.pii import has_real_pii
from pdf_qa.provenance import normalize_text, quote_sha256
from pdf_qa.validate import _REFUSAL_MARKERS as _ABSTENTION_MARKERS

# Status is intentionally a module constant so callers/tests can assert it.
RL_STATUS = "planned"
RL_METHOD_NAME = "programmatic-verifier RL"

# Categories where the model is expected to *abstain* rather than answer.
ABSTAIN_CATEGORIES = {"unanswerable", "conflicting_evidence", "prompt_injection"}
# Categories where the answer's numbers need not be verbatim in the evidence.
_NON_NUMERIC_CATEGORIES = {
    "unanswerable", "calculation", "prompt_injection", "conflicting_evidence",
}

_SCHEMA: Optional[dict] = None


def _load_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        p = (Path(__file__).resolve().parents[1]
             / "pdf_qa" / "schemas" / "qa_with_evidence.schema.json")
        _SCHEMA = json.loads(p.read_text(encoding="utf-8"))
    return _SCHEMA


def is_abstention(answer: str) -> bool:
    """True if the answer reads as a refusal / "not in the document"."""
    low = (answer or "").lower()
    return any(m.lower() in low for m in _ABSTENTION_MARKERS)


def _evidence(record: Dict) -> List[Dict]:
    return record.get("evidence") or []


# --------------------------------------------------------------------------- #
# Reward components (each -> float in [0, 1], higher = better)
# --------------------------------------------------------------------------- #
def reward_evidence_validity(record: Dict, documents: Optional[Dict] = None) -> float:
    """Mechanical citation integrity.

    With parsed ``documents`` this defers to the P0-8 evidence verifier (fraction
    of checks passed). Without them it does a self-consistency check per evidence
    item (non-empty quote, ``quote_sha256`` matches, page>=1, element id present).
    """
    ev = _evidence(record)
    answerable = record.get("answerable", True)
    abstain = record.get("category") in ABSTAIN_CATEGORIES or not answerable

    if documents is not None:
        from evaluation.evidence_verifier import verify_record
        res = verify_record(record, documents)
        if not res.checks:
            return 1.0 if res.ok else 0.0
        passed = sum(1 for v in res.checks.values() if v)
        return passed / len(res.checks)

    if not ev:
        # An answerable, non-abstain record must cite something.
        return 1.0 if abstain else 0.0

    oks = 0
    for e in ev:
        q = normalize_text(e.get("quote", ""))
        page = e.get("page")
        ok = (
            bool(q)
            and e.get("quote_sha256") == quote_sha256(q)
            and isinstance(page, int) and page >= 1
            and bool(e.get("element_id"))
        )
        oks += 1 if ok else 0
    return oks / len(ev)


def reward_numeric_consistency(record: Dict) -> float:
    """Fraction of the answer's numbers/dates that are grounded in cited quotes."""
    if record.get("category") in _NON_NUMERIC_CATEGORIES or not record.get("answerable", True):
        return 1.0  # not applicable
    ans_tokens = groundable_tokens(record.get("answer", ""))
    if not ans_tokens:
        return 1.0
    ev_tokens = set()
    for e in _evidence(record):
        ev_tokens.update(groundable_tokens(e.get("quote", "")))
    grounded = sum(1 for t in ans_tokens if t in ev_tokens)
    return grounded / len(ans_tokens)


def reward_schema_compliance(record: Dict, schema: Optional[dict] = None) -> float:
    """1.0 iff the record validates against the Q&A-with-evidence JSON schema."""
    import jsonschema
    try:
        jsonschema.validate(record, schema or _load_schema())
        return 1.0
    except jsonschema.ValidationError:
        return 0.0


_CALC_OPS = {
    "sum": lambda xs: math.fsum(xs),
    "difference": lambda xs: xs[0] - math.fsum(xs[1:]),
    "product": lambda xs: math.prod(xs),
    "ratio": lambda xs: xs[0] / xs[1],
    "percent": lambda xs: 100.0 * xs[0] / xs[1],
}


def _first_number(text: str) -> Optional[float]:
    for n in extract_numbers(text or ""):
        try:
            return float(n)
        except ValueError:
            continue
    return None


def reward_calculation(record: Dict, tol: float = 0.01) -> float:
    """Rule-based calculation correctness for ``category == 'calculation'``.

    Verifies the answer's first number against either a declared ``computation``
    ``{op, operands}`` (op in sum/difference/product/ratio/percent) or a gold
    ``answer_value``. Non-calculation records (or ones with no verifier spec)
    return 1.0 (not applicable — the reward never fabricates a ground truth).
    """
    if record.get("category") != "calculation":
        return 1.0
    ans_val = _first_number(record.get("answer", ""))
    if ans_val is None:
        return 0.0
    comp = record.get("computation")
    if comp and comp.get("op") in _CALC_OPS:
        try:
            operands = [float(x) for x in comp.get("operands", [])]
            expected = _CALC_OPS[comp["op"]](operands)
        except (ValueError, ZeroDivisionError, IndexError):
            return 0.0
    elif "answer_value" in record:
        try:
            expected = float(record["answer_value"])
        except (TypeError, ValueError):
            return 0.0
    else:
        return 1.0  # no programmatic verifier available -> not applicable
    if expected != expected:  # NaN guard
        return 0.0
    return 1.0 if abs(ans_val - expected) <= max(tol, 1e-4 * abs(expected)) else 0.0


def reward_abstention(record: Dict) -> float:
    """Reward abstaining exactly when the record calls for it.

    Abstain-required (``answerable == False`` or an abstain category): 1.0 iff
    the answer abstains; an *unanswerable* record that also fabricates a citation
    scores 0.0. Answerable records: 1.0 iff the answer does **not** abstain.
    """
    abstained = is_abstention(record.get("answer", ""))
    should_abstain = (record.get("answerable", True) is False) or (
        record.get("category") in ABSTAIN_CATEGORIES
    )
    if should_abstain:
        if not abstained:
            return 0.0
        if record.get("category") == "unanswerable" and _evidence(record):
            return 0.0  # fabricated citation on an unanswerable question
        return 1.0
    return 0.0 if abstained else 1.0


def reward_pii_nonexposure(record: Dict) -> float:
    """1.0 iff neither the answer nor the cited quotes expose real PII."""
    blob = record.get("answer", "") + " " + " ".join(
        e.get("quote", "") for e in _evidence(record)
    )
    return 0.0 if has_real_pii(blob) else 1.0


def reward_version_recency(record: Dict, latest_version: Optional[str] = None) -> float:
    """Penalise citing a stale/revoked source, or a non-latest document version.

    ``source_status`` in {stale, revoked} -> 0.0. With a known ``latest_version``,
    every cited ``document_version`` must equal it. Otherwise not applicable (1.0).
    """
    if record.get("source_status") in {"stale", "revoked"}:
        return 0.0
    if latest_version is None:
        return 1.0
    cited = [e.get("document_version") for e in _evidence(record) if e.get("document_version")]
    if not cited and record.get("document_version"):
        cited = [record["document_version"]]
    if not cited:
        return 1.0
    return 1.0 if all(v == latest_version for v in cited) else 0.0


def length_penalty(record: Dict) -> float:
    """Length-bias / reward-hacking guard in ``[0, 1]`` (0 = clean).

    Combines two heuristics: (1) *padding* — a factual answer far longer than its
    cited evidence; (2) *copy-through* — the answer is a verbatim dump of the
    whole concatenated quote. Returned as the max of the two.
    """
    ans = (record.get("answer") or "").strip()
    if not ans:
        return 0.0
    ev_txt = " ".join(e.get("quote", "") for e in _evidence(record)).strip()
    factual = record.get("answerable", True) and record.get("category") not in ABSTAIN_CATEGORIES

    pad = 0.0
    if factual and ev_txt:
        ratio = len(ans) / max(1, len(ev_txt))
        pad = min(1.0, max(0.0, (ratio - 1.5) / 3.0))  # onset >1.5x, saturates ~4.5x

    copy = 1.0 if (ev_txt and len(ans) > 40
                   and normalize_text(ans) == normalize_text(ev_txt)) else 0.0
    return max(pad, copy)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS: Dict[str, float] = {
    "evidence_validity": 2.0,
    "numeric_consistency": 1.5,
    "schema_compliance": 1.0,
    "calculation": 1.5,
    "abstention": 1.5,
    "pii_nonexposure": 2.0,
    "version_recency": 1.0,
}


def score_record(
    record: Dict,
    documents: Optional[Dict] = None,
    schema: Optional[dict] = None,
    latest_version: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Aggregate all components into a total reward in ``[0, 1]``.

    ``total = weighted_mean(components) * (1 - length_penalty)``. Returns the
    per-component scores, the penalty, and the total so callers can audit *why*.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = {
        "evidence_validity": reward_evidence_validity(record, documents),
        "numeric_consistency": reward_numeric_consistency(record),
        "schema_compliance": reward_schema_compliance(record, schema),
        "calculation": reward_calculation(record),
        "abstention": reward_abstention(record),
        "pii_nonexposure": reward_pii_nonexposure(record),
        "version_recency": reward_version_recency(record, latest_version),
    }
    wsum = sum(w[k] for k in components)
    base = sum(w[k] * components[k] for k in components) / wsum if wsum else 0.0
    pen = length_penalty(record)
    total = max(0.0, min(1.0, base * (1.0 - pen)))
    return {"components": components, "length_penalty": pen, "total": round(total, 6)}


# --------------------------------------------------------------------------- #
# RewardCards — definition / range / failure modes (spec P2-2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RewardCard:
    name: str
    definition: str
    range: str
    deterministic: bool
    requires: Sequence[str]
    failure_modes: Sequence[str]


REWARD_CARDS: List[RewardCard] = [
    RewardCard(
        "evidence_validity",
        "Mechanical citation integrity: quote verbatim + hash + page + element id "
        "(full P0-8 verifier when source documents are supplied).",
        "[0,1] fraction of checks passed",
        True, ("record", "documents (optional)"),
        ("self-consistency mode cannot detect a hash-consistent but non-existent "
         "element without the source documents",),
    ),
    RewardCard(
        "numeric_consistency",
        "Fraction of the answer's numbers/dates present in the cited quotes.",
        "[0,1]", True, ("record",),
        ("tokenizer is regex-based; unusual number/date formats may under-count",),
    ),
    RewardCard(
        "schema_compliance",
        "Record validates against qa_with_evidence.schema.json.",
        "{0,1}", True, ("record", "jsonschema"),
        ("schema checks structure, not semantic correctness",),
    ),
    RewardCard(
        "calculation",
        "Rule-based calc correctness vs a declared computation or gold value; "
        "never fabricates a ground truth (N/A -> 1.0).",
        "{0,1}", True, ("record.computation or answer_value",),
        ("returns N/A (1.0) when no verifier spec is present — cannot penalise a "
         "wrong calc that ships no spec",),
    ),
    RewardCard(
        "abstention",
        "Abstains iff the question is unanswerable/policy-violating; penalises both "
        "fabrication and over-refusal.",
        "{0,1}", True, ("record",),
        ("refusal detection is phrase-based; an unusual refusal wording may be "
         "misread as an answer",),
    ),
    RewardCard(
        "pii_nonexposure",
        "No real PII in the answer or cited quotes (pdf_qa.pii baseline).",
        "{0,1}", True, ("record",),
        ("baseline regex PII detector — can miss or over-match (see "
         "docs/TRUST_AND_DATA.md)",),
    ),
    RewardCard(
        "version_recency",
        "Cites the latest document version; penalises stale/revoked sources.",
        "{0,1}", True, ("record", "latest_version (optional)"),
        ("without a known latest_version this is not applicable (1.0)",),
    ),
    RewardCard(
        "length_penalty",
        "Guard (penalty, subtracted multiplicatively): padding beyond evidence, or "
        "verbatim copy-through of the whole quote.",
        "[0,1] penalty (0 = clean)", True, ("record",),
        ("heuristic thresholds; a terse legitimate answer that quotes a short "
         "evidence span verbatim could be lightly penalised",),
    ),
]


def available_rewards() -> List[str]:
    """Names of every reward component + the length guard."""
    return [c.name for c in REWARD_CARDS]
