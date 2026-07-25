"""Memoirist QA evaluation scorer (isolated from the ``pdf_qa`` core).

A repeatable, two-layer QC gate + regression eval for the memoirist dataset.
See :mod:`evaluation.qa_scorer` for the scoring engine and :mod:`evaluation.run_eval`
for the CLI. Detection rules live entirely in ``rubric.yaml``.
"""

from .qa_scorer import (
    DimResult,
    Judge,
    LLMJudge,
    PairScore,
    RecordingJudge,
    ReplayJudge,
    Rubric,
    StubJudge,
    aggregate,
    aggregate_by_chunk,
    check_first_person,
    check_format,
    check_leading_q,
    check_register,
    judge_key,
    load_pairs,
    load_rubric,
    normalize_pair,
    score_pairs,
    summarize_runs,
)

__all__ = [
    "DimResult",
    "Judge",
    "LLMJudge",
    "PairScore",
    "RecordingJudge",
    "ReplayJudge",
    "Rubric",
    "StubJudge",
    "aggregate",
    "aggregate_by_chunk",
    "check_first_person",
    "check_format",
    "check_leading_q",
    "check_register",
    "judge_key",
    "load_pairs",
    "load_rubric",
    "normalize_pair",
    "score_pairs",
    "summarize_runs",
]
