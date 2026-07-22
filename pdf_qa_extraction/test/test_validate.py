"""Tests for generated-pair quality control (``pdf_qa.validate``), dependency-free.

Covers validation (empty / too-short / q==a / refusal), exact + near-duplicate
removal, the config/kwargs precedence, and the stats breakdown.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pdf_qa.config import QAConfig
from pdf_qa.validate import clean_qa_pairs, format_stats, validate_pair


def test_validate_pair_reasons():
    assert validate_pair({"QUESTION": "정상적인 질문입니까?", "ANSWER": "정상적인 답변입니다."}) is None
    assert validate_pair({"QUESTION": "", "ANSWER": "답"}) == "empty"
    assert validate_pair({"ANSWER": "답만 있음"}) == "empty"
    assert validate_pair({"QUESTION": "x?", "ANSWER": "y"}) == "too_short"
    assert validate_pair({"QUESTION": "같은 문장입니다", "ANSWER": "같은 문장입니다"}) == "q_equals_a"
    assert (
        validate_pair(
            {"QUESTION": "3분기 성장률은?", "ANSWER": "제공된 정보로는 알 수 없습니다."}
        )
        == "refusal"
    )
    # A refusal-looking English answer is also dropped.
    assert (
        validate_pair({"QUESTION": "What is the value?", "ANSWER": "I don't know."})
        == "refusal"
    )


def test_validate_pair_respects_thresholds_and_flags():
    qa = {"QUESTION": "짧은질문", "ANSWER": "짧은답"}
    # Lower the minimums so the short pair passes.
    assert validate_pair(qa, min_question_chars=3, min_answer_chars=2) is None
    # Refusal filtering can be turned off.
    refusal = {"QUESTION": "값은 무엇입니까?", "ANSWER": "정보가 없습니다."}
    assert validate_pair(refusal) == "refusal"
    assert validate_pair(refusal, drop_refusals=False) is None


def test_clean_removes_invalid_and_reports_reasons():
    pairs = [
        {"QUESTION": "이 문서의 핵심 주제는 무엇입니까?", "ANSWER": "국제 금융 동향입니다."},
        {"QUESTION": "x", "ANSWER": "y"},                       # too short
        {"QUESTION": "", "ANSWER": "질문이 없습니다"},            # empty
        {"QUESTION": "값은 얼마입니까?", "ANSWER": "확인할 수 없습니다."},  # refusal
    ]
    cleaned, stats = clean_qa_pairs(pairs, QAConfig())
    assert [p["QUESTION"] for p in cleaned] == ["이 문서의 핵심 주제는 무엇입니까?"]
    assert stats["input"] == 4 and stats["kept"] == 1 and stats["removed"] == 3
    assert stats["reasons"]["too_short"] == 1
    assert stats["reasons"]["empty"] == 1
    assert stats["reasons"]["refusal"] == 1


def test_exact_duplicate_questions_removed_keeping_first():
    pairs = [
        {"QUESTION": "What is GDP?", "ANSWER": "First answer here."},
        {"QUESTION": "what is gdp",  "ANSWER": "Second, normalized dup."},
        {"QUESTION": "What is GDP??", "ANSWER": "Third, punctuation dup."},
    ]
    cleaned, stats = clean_qa_pairs(pairs, QAConfig())
    assert len(cleaned) == 1
    assert cleaned[0]["ANSWER"] == "First answer here."
    assert stats["reasons"]["duplicate"] == 2


def test_near_duplicate_detection_and_threshold_gate():
    qa = "What is the reported GDP growth rate for the third quarter of the year"
    pairs = [
        {"QUESTION": qa, "ANSWER": "It is 3.8 percent for the quarter."},
        {"QUESTION": qa + " exactly", "ANSWER": "A different answer body."},
    ]
    # Default 0.9 threshold catches the near-duplicate.
    cleaned, stats = clean_qa_pairs(pairs, QAConfig())
    assert len(cleaned) == 1 and stats["reasons"]["near_duplicate"] == 1
    # threshold 1.0 keeps only exact-dup removal, so both survive.
    cleaned2, stats2 = clean_qa_pairs(pairs, QAConfig(), similarity_threshold=1.0)
    assert len(cleaned2) == 2 and stats2["reasons"]["near_duplicate"] == 0


def test_disabling_validation_and_dedup_keeps_all_usable_pairs():
    pairs = [
        {"QUESTION": "q one?", "ANSWER": "a"},
        {"QUESTION": "q one?", "ANSWER": "a"},   # exact dup, but dedup off
        {"not": "a qa dict"},                     # still unusable -> dropped
    ]
    cleaned, stats = clean_qa_pairs(pairs, QAConfig(), validate=False, dedup=False)
    assert len(cleaned) == 2
    assert stats["reasons"]["empty"] == 1


def test_config_flags_disable_cleaning():
    cfg = QAConfig(validate_qa=False, dedup_qa=False)
    pairs = [
        {"QUESTION": "dup?", "ANSWER": "same"},
        {"QUESTION": "dup?", "ANSWER": "same"},
    ]
    cleaned, _ = clean_qa_pairs(pairs, cfg)
    assert len(cleaned) == 2  # dedup disabled via config


def test_format_stats_is_human_readable():
    _, stats = clean_qa_pairs(
        [{"QUESTION": "x", "ANSWER": "y"}], QAConfig()
    )
    summary = format_stats(stats)
    assert "입력 1개" in summary and "too_short=1" in summary
