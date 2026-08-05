"""P1-5 six-arm runner tests: the CPU-testable harness (retrieval → prompt items →
prediction assembly → shared-contract scoring → aggregate-from-raw → paired bootstrap →
artifact writing) is exercised end-to-end with a stub generator (no model download), and
the train corpus is proven leakage-safe against the eval set.

These validate orchestration correctness; the actual model numbers are produced by the
GPU run (INT4 arms are honestly ``not_measured`` without CUDA)."""

import json
import os

import pytest

pytest.importorskip("yaml")  # quantization.data_korquad (imported lazily by the runner)

from evaluation.pdf_native import assert_no_split_leakage
from workflows.pdf_native_post_training.benchmarks.pdf_native import run_arms as RA


# --------------------------------------------------------------------------- fixtures
def _eval_rows():
    return RA.load_eval_rows()


def _perfect(rows):
    gold = {r["question"]: r.get("answer", "") for r in rows}
    return lambda items: [gold.get(it["question"], "") for it in items]


def _empty(items):
    return ["" for _ in items]


def _retriever(rows):
    corpus = RA.build_eval_corpus(rows)
    idx = RA.BM25Index.build(corpus)
    return RA.Retriever(idx), {c["element_id"]: c for c in corpus}, idx


# --------------------------------------------------------------------------- data loaders
def test_pii_terms_are_the_canary_values():
    terms = RA.load_pii_terms()
    assert "canary@example.com" in terms
    assert any(t.startswith("4111-") for t in terms)  # the Luhn-invalid card canary
    assert "email" not in terms and "canaries" not in terms  # keys must not leak in as terms


def test_eval_corpus_contains_no_gold_answers():
    rows = _eval_rows()
    corpus = RA.build_eval_corpus(rows)
    assert corpus, "retrieval corpus should not be empty"
    joined = "\n".join(c["text"] for c in corpus)
    # the retrieval corpus is built from document evidence quotes only; gold answer
    # strings must never be indexed (that would leak labels into retrieval)
    for r in rows:
        ans = (r.get("answer") or "").strip()
        if ans and r.get("answerable"):
            assert ans not in joined, f"gold answer leaked into retrieval corpus: {ans}"
    # every corpus doc corresponds to a real evidence element id
    ev_ids = {e["element_id"] for r in rows for e in (r.get("evidence") or [])}
    assert {c["element_id"] for c in corpus} <= ev_ids


# --------------------------------------------------------------------------- harness
def test_perfect_predictions_score_high_and_reproduce_from_raw():
    rows = _eval_rows()
    gen = RA.generate_predictions(rows, _perfect(rows))
    scored = RA.score_arm(rows, gen, pii_terms=RA.load_pii_terms())
    agg = scored["aggregate"]
    assert agg["f1"] > 0.9 and agg["em"] > 0.8
    assert agg["pii_leakage_rate"] == 0.0
    # the aggregate MUST be reproducible purely from the per-example records (completion gate)
    regen = RA.aggregate_from_per_example(scored["per_example"])
    assert regen["f1"] == agg["f1"] and regen["em"] == agg["em"]


def test_empty_predictions_score_low():
    rows = _eval_rows()
    gen = RA.generate_predictions(rows, _empty)
    agg = RA.score_arm(rows, gen, pii_terms=RA.load_pii_terms())["aggregate"]
    assert (agg["f1"] or 0.0) < 0.2


def test_retrieval_arm_records_recall_and_citations():
    rows = _eval_rows()
    retr, cbi, _ = _retriever(rows)
    gen = RA.generate_predictions(rows, _perfect(rows), retriever=retr, corpus_by_id=cbi, k=4)
    scored = RA.score_arm(rows, gen, pii_terms=RA.load_pii_terms(), k=4)
    assert "retrieval_recall_at_k" in scored["aggregate"]
    # at least some answerable rows carry retrieved ids
    assert any(gen["retrieved_map"].get(r["qa_id"]) for r in rows if r.get("answerable"))


def test_paired_bootstrap_sign_and_significance():
    rows = _eval_rows()
    good = RA.score_arm(rows, RA.generate_predictions(rows, _perfect(rows)),
                        pii_terms=[])["per_example"]
    bad = RA.score_arm(rows, RA.generate_predictions(rows, _empty), pii_terms=[])["per_example"]
    d = RA.paired_bootstrap_delta(good, bad, "f1")
    assert d["delta"] > 0 and d["significant"] is True and d["n_pairs"] > 0
    # ordering flips sign
    d2 = RA.paired_bootstrap_delta(bad, good, "f1")
    assert d2["delta"] < 0


def test_paired_bootstrap_no_pairs_is_not_measured():
    d = RA.paired_bootstrap_delta([], [], "f1")
    assert d["delta"] == "not_measured" and d["n_pairs"] == 0


# --------------------------------------------------------------------------- finalize/artifacts
def _finalize_two_arms(tmp_path, rows):
    retr, cbi, idx = _retriever(rows)
    per_ex_dir = os.path.join(tmp_path, "per_example")
    os.makedirs(per_ex_dir)
    results, store = {}, {}
    for arm, fn in (("base_bf16", _empty), ("sft_bf16", _perfect(rows))):
        sc = RA.score_arm(rows, RA.generate_predictions(rows, fn), pii_terms=[])
        RA._write_per_example(per_ex_dir, arm, 42, sc["per_example"])
        store[f"{arm}@42"] = sc["per_example"]
        agg = dict(sc["aggregate"]); agg.update(size_gb="not_measured", peak_vram_gb="not_measured")
        results.setdefault(arm, []).append({"seed": 42, **agg})
    for arm, fn in (("base_bf16_retrieval", _empty), ("sft_bf16_retrieval", _perfect(rows))):
        sc = RA.score_arm(rows, RA.generate_predictions(rows, fn, retriever=retr,
                          corpus_by_id=cbi, k=4), pii_terms=[], k=4)
        RA._write_per_example(per_ex_dir, arm, 42, sc["per_example"])
        store[f"{arm}@42"] = sc["per_example"]
        agg = dict(sc["aggregate"]); agg.update(size_gb="not_measured", peak_vram_gb="not_measured")
        results.setdefault(arm, []).append({"seed": 42, **agg})
    cfg = RA.make_config("stub", smoke=True)
    return RA._finalize(tmp_path, "stub", [42], results, store, idx, cfg, smoke=True, cuda=False)


def test_finalize_writes_honest_artifacts(tmp_path):
    rows = _eval_rows()
    m = _finalize_two_arms(str(tmp_path), rows)
    assert os.path.exists(os.path.join(tmp_path, "summary.json"))
    assert os.path.exists(os.path.join(tmp_path, "report.md"))
    # INT4 arms cannot run without CUDA -> honestly marked not_measured
    for arm in ("sft_int4_ptq", "sft_int4_qat"):
        assert m["arms"][arm]["status"] == "not_measured"
        assert m["arms"][arm]["reason"] == "requires_cuda"
    # the two spec-critical comparisons are present with a positive, significant SFT effect
    comp = m["comparisons"]["sft_vs_base_closed_book"]["f1"]
    assert comp["delta"] > 0 and comp["significant"] is True
    # a smoke run must be clearly flagged as not publishable
    assert "WARNING" in m and "SMOKE" in m["WARNING"]
    # summary.json must be valid JSON and carry provenance
    data = json.load(open(os.path.join(tmp_path, "summary.json"), encoding="utf-8"))
    assert data["retriever"]["kind"] == "bm25"
    assert data["eval_set_sha256"] and data["eval_system_prompt_sha256"]


# --------------------------------------------------------------------------- train-corpus leakage
def test_train_corpus_is_leakage_safe_vs_eval():
    train_path = os.path.join(os.path.dirname(RA.__file__), "train", "train.jsonl")
    if not os.path.exists(train_path):
        pytest.skip("train corpus not generated")
    with open(train_path, encoding="utf-8") as fh:
        train = [json.loads(l) for l in fh if l.strip()]
    audit = assert_no_split_leakage({"train": train, "eval": _eval_rows()})
    assert audit["disjoint"] is True
    # families are genuinely disjoint sets
    fams_train = set(audit["families"]["train"])
    fams_eval = set(audit["families"]["eval"])
    assert fams_train and fams_eval and not (fams_train & fams_eval)
