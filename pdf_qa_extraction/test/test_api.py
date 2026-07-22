"""Tests for the one-call facade ``extract_qa`` (no heavy deps).

The pipeline internals are monkeypatched so these run in a bare env; they verify
that env is the baseline, kwargs override it, ``out`` routes through
``run_pipeline``, and a provider is built only when no ``provider_obj`` is given.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import pdf_qa.api as api
from pdf_qa import extract_qa


def test_returns_pairs_and_passes_provider_obj(monkeypatch):
    seen = {}

    def fake_gen(pdf, llm, config):
        seen.update(pdf=pdf, llm=llm, config=config)
        return [{"QUESTION": "What is the topic?", "ANSWER": "The topic is finance."}]

    monkeypatch.setattr(api, "generate_qa_pairs", fake_gen)
    sentinel = object()
    out = extract_qa("r.pdf", provider_obj=sentinel)

    # The facade returns validated + de-duplicated pairs.
    assert out == [{"QUESTION": "What is the topic?", "ANSWER": "The topic is finance."}]
    assert seen["pdf"] == "r.pdf"
    assert seen["llm"] is sentinel
    # env baseline still yields a valid config (persona default present).
    assert seen["config"].persona


def test_kwargs_override_env(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        api, "generate_qa_pairs", lambda pdf, llm, cfg: seen.update(cfg=cfg) or []
    )

    extract_qa(
        "r.pdf",
        provider_obj=object(),
        persona="feynman",
        domain="Constitutional Law",
        num_questions="9",
        num_img_questions="3",
        strategy="hi_res",
        gpu_boost=True,
        table_model="detectron2",
    )
    cfg = seen["cfg"]
    assert cfg.persona == "feynman"
    assert cfg.domain == "Constitutional Law"
    assert cfg.num_questions == "9"
    assert cfg.num_img_questions == "3"
    assert cfg.strategy == "hi_res"
    assert cfg.gpu_boost is True
    assert cfg.table_model == "detectron2"


def test_out_path_routes_through_run_pipeline(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(
        api,
        "run_pipeline",
        lambda pdf, out, llm, cfg: calls.update(pdf=pdf, out=out) or [{"x": 1}],
    )

    def _boom(*a, **k):
        raise AssertionError("generate_qa_pairs must not run when out= is set")

    monkeypatch.setattr(api, "generate_qa_pairs", _boom)

    dest = str(tmp_path / "o.jsonl")
    res = extract_qa("r.pdf", out=dest, provider_obj=object())
    assert res == [{"x": 1}]
    assert calls["out"] == dest


def test_builds_provider_when_no_provider_obj(monkeypatch):
    picked = {}

    def fake_get_provider(name, config=None):
        picked["name"] = name
        return "LLM"

    monkeypatch.setattr(api, "get_provider", fake_get_provider)
    monkeypatch.setattr(
        api,
        "generate_qa_pairs",
        lambda pdf, llm, cfg: [
            {"QUESTION": "Which backend?", "ANSWER": "The selected one.", "llm": llm}
        ],
    )

    res = extract_qa("r.pdf", provider="ollama")
    assert picked["name"] == "ollama"
    # The built provider object was threaded into generation; pairs are curated.
    assert res == [
        {"QUESTION": "Which backend?", "ANSWER": "The selected one.", "llm": "LLM"}
    ]
