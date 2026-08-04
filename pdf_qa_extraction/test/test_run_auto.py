"""Tests for the zero-config batch runner ``run_auto.py``.

The pipeline is stubbed (``get_provider`` + ``generate_qa_pairs``) so the batch
orchestration is exercised without unstructured or a live LLM. Beyond the basic
per-file / combined / manifest wiring these tests pin the P0-10 correctness
contract: content+config-hash skipping, aggregate retention of cached docs,
corruption invalidation, and non-zero exit on failure.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pdf_qa
import pdf_qa.pipeline as pipeline


class _Dummy:
    name = "dummy"


def _pairs():
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


def _fake_gen(pdf, llm, config):
    return _pairs()


def _prepare(tmp_path, monkeypatch, n=2, gen=_fake_gen):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    for i in range(n):
        (in_dir / f"doc_{i}.pdf").write_bytes(b"%PDF-1.4 fake " + str(i).encode())
    monkeypatch.setenv("INPUT_DIR", str(in_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OVERWRITE", raising=False)
    monkeypatch.delenv("ALLOW_PARTIAL", raising=False)
    monkeypatch.delenv("PERSONA", raising=False)
    monkeypatch.setattr(pdf_qa, "get_provider", lambda *a, **k: _Dummy())
    monkeypatch.setattr(pipeline, "generate_qa_pairs", gen)
    return in_dir, out_dir


def _manifest(out_dir):
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def test_batches_all_pdfs_and_writes_manifest(tmp_path, monkeypatch):
    _, out_dir = _prepare(tmp_path, monkeypatch, n=2)
    import run_auto

    assert run_auto.main() == 0
    assert (out_dir / "doc_0.qa.jsonl").exists()
    assert (out_dir / "doc_1.qa.jsonl").exists()
    assert (out_dir / "all.qa.jsonl").exists()

    lines = (out_dir / "doc_0.qa.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["QUESTION"] == "이 문서의 핵심 주제는 무엇입니까?"

    manifest = _manifest(out_dir)
    assert manifest["totals"]["documents"] == 2
    assert manifest["totals"]["pairs"] == 4
    assert manifest["totals"]["figures"] == 2
    assert manifest["status"] == "ok"
    docs = [d for d in manifest["documents"] if d.get("figures")]
    assert docs and docs[0]["figures"][0]["section"] == "Intro"
    assert docs[0]["figures"][0]["context_used"] is True


def test_idempotent_skip_and_overwrite(tmp_path, monkeypatch):
    _, out_dir = _prepare(tmp_path, monkeypatch, n=1)
    import run_auto

    run_auto.main()  # first pass writes doc_0.qa.jsonl
    run_auto.main()  # second pass must skip it (unchanged contract)
    manifest = _manifest(out_dir)
    assert any(d.get("cached") for d in manifest["documents"])
    assert manifest["counts_run"]["cached"] == 1

    monkeypatch.setenv("OVERWRITE", "1")
    run_auto.main()
    manifest = _manifest(out_dir)
    assert manifest["totals"]["documents"] == 1
    assert not any(d.get("cached") for d in manifest["documents"])
    assert manifest["counts_run"]["processed"] == 1


def test_incremental_add_keeps_existing_in_aggregate(tmp_path, monkeypatch):
    in_dir, out_dir = _prepare(tmp_path, monkeypatch, n=2)
    import run_auto

    assert run_auto.main() == 0
    # Add a third document and re-run: the first two must be cached, not dropped.
    (in_dir / "doc_2.pdf").write_bytes(b"%PDF-1.4 fake 2")
    assert run_auto.main() == 0

    manifest = _manifest(out_dir)
    assert manifest["counts_run"] == {"processed": 1, "cached": 2, "failed": 0}
    assert manifest["totals"]["documents"] == 3
    combined = (out_dir / "all.qa.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(combined) == 6  # 3 docs x 2 pairs — nothing dropped


def test_changed_content_same_name_invalidates(tmp_path, monkeypatch):
    in_dir, out_dir = _prepare(tmp_path, monkeypatch, n=1)
    import run_auto

    run_auto.main()
    assert _manifest(out_dir)["counts_run"]["processed"] == 1
    # Same filename, different bytes -> content hash changes -> reprocess.
    (in_dir / "doc_0.pdf").write_bytes(b"%PDF-1.4 DIFFERENT CONTENT")
    run_auto.main()
    m = _manifest(out_dir)
    assert m["counts_run"]["processed"] == 1 and m["counts_run"]["cached"] == 0


def test_persona_change_invalidates(tmp_path, monkeypatch):
    _, out_dir = _prepare(tmp_path, monkeypatch, n=1)
    import run_auto

    run_auto.main()
    assert _manifest(out_dir)["counts_run"]["cached"] == 0
    run_auto.main()
    assert _manifest(out_dir)["counts_run"]["cached"] == 1  # unchanged -> cached
    # Change the persona: the resolved config + prompt ledger change -> reprocess.
    monkeypatch.setenv("PERSONA", "socratic")
    run_auto.main()
    assert _manifest(out_dir)["counts_run"]["cached"] == 0


def test_truncated_output_is_invalidated(tmp_path, monkeypatch):
    _, out_dir = _prepare(tmp_path, monkeypatch, n=1)
    import run_auto

    run_auto.main()
    # Corrupt the cached JSONL (truncated JSON line) -> must not be a cache hit.
    (out_dir / "doc_0.qa.jsonl").write_text('{"QUESTION": "broke', encoding="utf-8")
    run_auto.main()
    m = _manifest(out_dir)
    assert m["counts_run"]["processed"] == 1 and m["counts_run"]["cached"] == 0
    # And the reprocessed file is valid ndjson again.
    lines = (out_dir / "doc_0.qa.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and all(json.loads(x) for x in lines)


def test_document_failure_exits_nonzero(tmp_path, monkeypatch):
    def _gen_with_failure(pdf, llm, config):
        if "doc_1" in os.path.basename(pdf):
            raise RuntimeError("provider exploded")
        return _pairs()

    _, out_dir = _prepare(tmp_path, monkeypatch, n=2, gen=_gen_with_failure)
    import run_auto

    rc = run_auto.main()
    assert rc == 1  # a failed document must not yield exit 0
    m = _manifest(out_dir)
    assert m["status"] == "failed"
    assert m["missing_documents"] == ["doc_1.pdf"]
    assert m["counts_run"] == {"processed": 1, "cached": 0, "failed": 1}
    failures = json.loads((out_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures["count"] == 1 and "provider exploded" in failures["failures"][0]["error"]
    # The healthy document is still exported.
    assert (out_dir / "doc_0.qa.jsonl").exists()

    # --allow-partial downgrades to a partial success (exit 0) but records it.
    assert run_auto.main(["--allow-partial"]) == 0
    m = _manifest(out_dir)
    assert m["status"] == "partial" and m["failure_rate"] == 0.5


def test_concurrent_run_is_refused(tmp_path, monkeypatch):
    _, out_dir = _prepare(tmp_path, monkeypatch, n=1)
    import run_auto

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / run_auto._LOCK_NAME).write_text("999999 9999999999\n", encoding="utf-8")
    # A fresh lock held by "another" process -> refuse with a non-zero exit.
    assert run_auto.main() == 2
    # The runner must not have clobbered outputs.
    assert not (out_dir / "doc_0.qa.jsonl").exists()
