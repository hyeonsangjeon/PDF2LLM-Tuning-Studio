"""Generated-pair quality control: validation + de-duplication (pure).

Turns a raw list of ``{"QUESTION": ..., "ANSWER": ...}`` dicts into a curated
fine-tuning dataset by

1. **validating** each pair -- dropping empty, too-short, question==answer and
   refusal / "not in the context" answers, and
2. **de-duplicating** -- removing exact-duplicate questions and (optionally)
   near-duplicates by token Jaccard similarity, keeping the first occurrence.

It is dependency-free (like :mod:`pdf_qa.manifest`) so the pipeline, the batch
runner and the web app can all share it. Everything is tunable through
:class:`~pdf_qa.config.QAConfig` (or explicit keyword arguments) and returns a
compact ``stats`` dict so callers can report exactly what was removed and why.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set, Tuple

# The generators emit UPPER-CASE keys; accept common variants defensively so a
# provider that returns lower-case keys is still validated rather than dropped.
_QUESTION_KEYS = ("QUESTION", "question", "Question", "q", "Q")
_ANSWER_KEYS = ("ANSWER", "answer", "Answer", "a", "A")

# Substrings that mark a non-answer (model refusal / "not in the context"),
# Korean + English. Matched case-insensitively as substrings.
_REFUSAL_MARKERS = (
    "cannot answer", "can't answer", "cannot determine", "not enough information",
    "no information", "not provided", "not mentioned", "unable to", "i don't know",
    "i do not know", "not applicable", "cannot be determined",
    "제공된 정보", "정보가 없", "알 수 없", "답변할 수 없", "답할 수 없", "확인할 수 없",
    "제공되지 않", "언급되지 않", "나와 있지 않", "찾을 수 없", "모르겠",
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def get_field(qa: dict, keys: Iterable[str]) -> str:
    """Return the first non-empty string value among ``keys`` (stripped)."""
    for key in keys:
        value = qa.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def question_of(qa: dict) -> str:
    """The question text of a pair (empty string if absent)."""
    return get_field(qa, _QUESTION_KEYS)


def answer_of(qa: dict) -> str:
    """The answer text of a pair (empty string if absent)."""
    return get_field(qa, _ANSWER_KEYS)


def _norm_key(text: str) -> str:
    """Normalise text for exact-duplicate detection.

    Lower-cases, strips punctuation and collapses whitespace so
    ``"What is GDP?"`` and ``"what is gdp"`` map to the same key.
    """
    text = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


def _tokens(text: str) -> Set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _looks_like_refusal(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def validate_pair(
    qa: dict,
    *,
    min_question_chars: int = 6,
    min_answer_chars: int = 4,
    drop_refusals: bool = True,
) -> Optional[str]:
    """Return a rejection reason for an invalid pair, or ``None`` if it passes.

    Reasons: ``"empty"`` (missing Q or A), ``"too_short"``, ``"q_equals_a"``,
    ``"refusal"``.
    """
    if not isinstance(qa, dict):
        return "empty"
    question = question_of(qa)
    answer = answer_of(qa)
    if not question or not answer:
        return "empty"
    if len(question) < min_question_chars or len(answer) < min_answer_chars:
        return "too_short"
    if _norm_key(question) == _norm_key(answer):
        return "q_equals_a"
    if drop_refusals and (_looks_like_refusal(question) or _looks_like_refusal(answer)):
        return "refusal"
    return None


def _resolve(value, config, attr, default):
    """Explicit kwarg wins; else the config attribute; else the default."""
    if value is not None:
        return value
    if config is not None:
        return getattr(config, attr, default)
    return default


def clean_qa_pairs(
    pairs: List[dict],
    config=None,
    *,
    validate: Optional[bool] = None,
    dedup: Optional[bool] = None,
    min_question_chars: Optional[int] = None,
    min_answer_chars: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    drop_refusals: Optional[bool] = None,
) -> Tuple[List[dict], dict]:
    """Validate + de-duplicate ``pairs`` and return ``(cleaned, stats)``.

    Options come from ``config`` (a :class:`~pdf_qa.config.QAConfig`) unless
    overridden by an explicit keyword. ``stats`` records ``input``/``kept``/
    ``removed`` counts and a per-reason breakdown, so callers can surface the QC
    result (pipeline log, web-app response, batch manifest).
    """
    do_validate = _resolve(validate, config, "validate_qa", True)
    do_dedup = _resolve(dedup, config, "dedup_qa", True)
    min_q = _resolve(min_question_chars, config, "min_question_chars", 6)
    min_a = _resolve(min_answer_chars, config, "min_answer_chars", 4)
    threshold = _resolve(similarity_threshold, config, "dedup_similarity", 0.9)
    refusals = _resolve(drop_refusals, config, "drop_refusals", True)

    reasons = {
        "empty": 0,
        "too_short": 0,
        "q_equals_a": 0,
        "refusal": 0,
        "duplicate": 0,
        "near_duplicate": 0,
    }
    kept: List[dict] = []
    seen_keys: Set[str] = set()
    kept_tokens: List[Set[str]] = []
    # Near-duplicate scanning is O(n^2); only run it when enabled + meaningful.
    near_dup = do_dedup and 0.0 < float(threshold) < 1.0

    for qa in pairs:
        if do_validate:
            reason = validate_pair(
                qa,
                min_question_chars=min_q,
                min_answer_chars=min_a,
                drop_refusals=refusals,
            )
            if reason:
                reasons[reason] += 1
                continue
        elif not isinstance(qa, dict) or not question_of(qa) or not answer_of(qa):
            # Even with validation off, a pair with no usable Q/A can't be
            # de-duplicated or written meaningfully — drop it as empty.
            reasons["empty"] += 1
            continue

        if do_dedup:
            question = question_of(qa)
            key = _norm_key(question)
            if key in seen_keys:
                reasons["duplicate"] += 1
                continue
            if near_dup:
                tokens = _tokens(question)
                if any(_jaccard(tokens, prev) >= threshold for prev in kept_tokens):
                    reasons["near_duplicate"] += 1
                    continue
                kept_tokens.append(tokens)
            seen_keys.add(key)

        kept.append(qa)

    removed = sum(reasons.values())
    stats = {
        "input": len(pairs),
        "kept": len(kept),
        "removed": removed,
        "reasons": reasons,
    }
    return kept, stats


def format_stats(stats: dict) -> str:
    """One-line human summary of a :func:`clean_qa_pairs` stats dict."""
    reasons = stats.get("reasons", {})
    detail = ", ".join(f"{name}={count}" for name, count in reasons.items() if count)
    base = (
        f"입력 {stats.get('input', 0)}개 → 유지 {stats.get('kept', 0)}개 "
        f"(제거 {stats.get('removed', 0)}개"
    )
    return f"{base}: {detail})" if detail else f"{base})"
