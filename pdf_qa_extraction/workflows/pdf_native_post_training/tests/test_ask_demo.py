"""Tests for ``pdf2llm ask`` (workflows.pdf_native_post_training.ask_demo).

Offline path only: replays the committed real A100 per-example outputs. Pure CPU,
no network, no GPU. Guards that the "짠!" demo stays honest — the answers it prints
are exactly the committed raw predictions, and retrieval genuinely lifts F1.
"""
import os

import pytest

from workflows.pdf_native_post_training import ask_demo

_PER = ask_demo._PER_EXAMPLE


def _has_data() -> bool:
    return os.path.isdir(_PER) and bool(os.listdir(_PER))


pytestmark = pytest.mark.skipif(not _has_data(), reason="historical per_example not present")


def test_default_question_prints_and_exits_zero(capsys):
    rc = ask_demo.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2024년 연간 매출액은 얼마입니까?" in out  # default q000
    assert "짠" in out
    assert "1,250억 원" in out  # gold + retrieval prediction surfaced


def test_all_six_arms_present_for_default():
    all_arms = ask_demo._load_all(42)
    assert {a for a, _ in ask_demo._ALL_ARMS} == set(all_arms)
    for arm, _ in ask_demo._ALL_ARMS:
        assert ask_demo._DEFAULT_QID in all_arms[arm], arm


def test_retrieval_beats_closed_book_on_default():
    by_arm = {arm: rows.get(ask_demo._DEFAULT_QID, {}) for arm, rows in ask_demo._load_all(42).items()}
    best_closed = max(ask_demo._f1_of(by_arm[a]) for a, _ in ask_demo._CLOSED_BOOK)
    best_open = max(ask_demo._f1_of(by_arm[a]) for a, _ in ask_demo._RETRIEVAL)
    assert best_open >= best_closed + 0.1  # the benchmark's core lesson


def test_printed_answer_matches_committed_raw(capsys):
    """The demo must not fabricate — the SFT+retrieval answer it prints is the
    exact committed prediction for that arm/seed/qid."""
    ask_demo.main(["--qa-id", "q000"])
    out = capsys.readouterr().out
    raw = ask_demo._load_arm(ask_demo._BEST_ARM, 42)["q000"]["_pred"]
    assert raw and raw in out


def test_unanswerable_handles_none_f1(capsys):
    rc = ask_demo.main(["--qa-id", "q014"])  # unanswerable → f1 is None
    out = capsys.readouterr().out
    assert rc == 0
    assert "기권" in out  # abstention framing, no crash on None


def test_substring_match_resolves_question(capsys):
    rc = ask_demo.main(["-q", "영업이익률"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "영업이익률은 몇 퍼센트입니까?" in out


def test_unknown_question_returns_2(capsys):
    rc = ask_demo.main(["-q", "존재하지않는질문xyz"])
    assert rc == 2


def test_list_covers_full_catalog(capsys):
    rc = ask_demo.main(["--list"])
    out = capsys.readouterr().out
    assert rc == 0
    catalog = ask_demo._question_catalog(42)
    assert len(catalog) >= 30
    for c in catalog:
        assert c["qa_id"] in out


def test_launcher_exposes_ask_subcommand():
    from pdf_qa.cli import build_parser

    args = build_parser().parse_args(["ask", "-q", "매출"])
    assert args.command == "ask"
    assert args.question == "매출"
