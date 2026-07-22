"""Tests for the zero-config batch runner ``run_auto.py``.

The pipeline is stubbed (``get_provider`` + ``generate_qa_pairs``) so the batch
orchestration — per-file JSONL, combined ``all.qa.jsonl``, ``manifest.json`` with
per-document figure linkage, and OVERWRITE idempotency — is exercised without
unstructured or a live LLM.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pdf_qa
import pdf_qa.pipeline as pipeline


class _Dummy:
    name = "dummy"


def _fake_gen(pdf, llm, config):
    return [
        {
            "QUESTION": "이 문서의 핵심 주제는 무엇입니까?",
            "ANSWER": "이 문서의 핵심 주제는 국제 금융 시장의 동향입니다.",
            "source": "text",
        },
        {
            "QUESTION": "차트에서 3분기 성장률은 얼마입니까?",
            "ANSWER": "차트에서 3분기 성장률은 3.8%입니다.",
            "source": "image",
            "image_path": "/f/fig-1.png",
            "page": 1,
            "section": "Intro",
            "figure_index": 1,
            "context_used": True,
        },
    ]


def _prepare(tmp_path, monkeypatch, n=2):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    for i in range(n):
        (in_dir / f"doc_{i}.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setenv("INPUT_DIR", str(in_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OVERWRITE", raising=False)
    monkeypatch.setattr(pdf_qa, "get_provider", lambda *a, **k: _Dummy())
    monkeypatch.setattr(pipeline, "generate_qa_pairs", _fake_gen)
    return in_dir, out_dir


def test_batches_all_pdfs_and_writes_manifest(tmp_path, monkeypatch):
    _, out_dir = _prepare(tmp_path, monkeypatch, n=2)
    import run_auto

    assert run_auto.main() == 0
    assert (out_dir / "doc_0.qa.jsonl").exists()
    assert (out_dir / "doc_1.qa.jsonl").exists()
    assert (out_dir / "all.qa.jsonl").exists()

    # per-file JSONL is valid ndjson
    lines = (out_dir / "doc_0.qa.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["QUESTION"] == "이 문서의 핵심 주제는 무엇입니까?"

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["totals"]["documents"] == 2
    assert manifest["totals"]["pairs"] == 4
    assert manifest["totals"]["figures"] == 2
    docs = [d for d in manifest["documents"] if d.get("figures")]
    assert docs and docs[0]["figures"][0]["section"] == "Intro"
    assert docs[0]["figures"][0]["context_used"] is True


def test_idempotent_skip_and_overwrite(tmp_path, monkeypatch):
    _, out_dir = _prepare(tmp_path, monkeypatch, n=1)
    import run_auto

    run_auto.main()  # first pass writes doc_0.qa.jsonl
    run_auto.main()  # second pass must skip it
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert any(d.get("skipped") for d in manifest["documents"])

    # OVERWRITE=1 reprocesses.
    monkeypatch.setenv("OVERWRITE", "1")
    run_auto.main()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["totals"]["documents"] == 1
    assert not any(d.get("skipped") for d in manifest["documents"])
