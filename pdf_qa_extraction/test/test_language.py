"""Tests for the output-language lock (``pdf_qa.prompts`` + ``QAConfig``).

Dependency-free: only exercises prompt rendering and config parsing, so no
LLM/unstructured stack is needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pdf_qa.config import QAConfig
from pdf_qa.prompts import (
    build_image_instruction,
    build_text_prompt,
    language_directive,
    resolve_language,
)


def test_resolve_language_auto_aliases():
    for value in ("", "auto", "source", "same", "match", None, "자동"):
        assert resolve_language(value) == "auto"


def test_resolve_language_known_codes_and_passthrough():
    assert resolve_language("ko") == "Korean"
    assert resolve_language("KOREAN") == "Korean"
    assert resolve_language("en") == "English"
    assert resolve_language("japanese") == "Japanese"
    # An unknown language name is passed through as given.
    assert resolve_language("Portuguese") == "Portuguese"


def test_text_prompt_auto_matches_source_and_has_no_leftover_placeholder():
    out = build_text_prompt("some English context", "Finance", "3", "professor", "auto")
    assert "{language_directive}" not in out
    assert "LANGUAGE LOCK" in out
    assert "SAME language as the source" in out


def test_text_prompt_forces_named_language():
    out = build_text_prompt("컨텍스트", "Finance", "3", "professor", "english")
    assert "written in English" in out
    assert "MUST be written in English" in out
    # The Korean few-shot examples are still present as a FORMAT illustration...
    assert "옵티머스" in out
    # ...but the prompt explicitly tells the model to ignore their language.
    assert "FORMAT ONLY" in out


def test_image_prompt_language_lock_and_context_coexist():
    out = build_image_instruction(
        "Finance", "2", "professor", context="[Section] GDP", language="korean"
    )
    assert "{language_directive}" not in out
    assert "written in Korean" in out
    assert "FIGURE CONTEXT" in out
    # The strict data-accuracy rule must survive alongside the language lock.
    assert "read from the image" in out.lower()


def test_language_directive_auto_vs_named():
    assert "detect" in language_directive("auto").lower()
    assert language_directive("ko") == "Korean"


def test_config_reads_output_language_env(monkeypatch):
    monkeypatch.delenv("QA_LANGUAGE", raising=False)
    monkeypatch.delenv("LANGUAGE", raising=False)
    monkeypatch.setenv("OUTPUT_LANGUAGE", "english")
    assert QAConfig.from_env().language == "english"
    # Default is auto when nothing is set.
    monkeypatch.delenv("OUTPUT_LANGUAGE", raising=False)
    assert QAConfig.from_env().language == "auto"


def test_config_language_alias(monkeypatch):
    monkeypatch.delenv("OUTPUT_LANGUAGE", raising=False)
    monkeypatch.delenv("LANGUAGE", raising=False)
    monkeypatch.setenv("QA_LANGUAGE", "japanese")
    assert QAConfig.from_env().language == "japanese"
