"""P1-5 benchmark-assembly tests: the public frozen regression set is consistent with
its source corpus, the leakage audit holds on the real fixture, `sealed`/`unseen` is
never claimed, and acceptance is pre-registered. Pure CPU."""

import hashlib
import json
import os

import pytest

yaml = pytest.importorskip("yaml")

_BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIR = os.path.join(_BENCH, "benchmarks", "pdf_native")
_CORPUS = os.path.join(_BENCH, "public_finance_demo")


def _load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _regression():
    return _load_jsonl(os.path.join(_DIR, "public_regression.jsonl"))


def _manifest():
    with open(os.path.join(_DIR, "final_manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- presence
def test_required_artifacts_exist():
    for f in ["benchmark.yaml", "acceptance.yaml", "public_regression.jsonl",
              "final_manifest.json", "checksums.sha256", "build_benchmark.py"]:
        assert os.path.exists(os.path.join(_DIR, f)), f
    assert os.path.isdir(os.path.join(_DIR, "splits"))


# --------------------------------------------------------------------------- consistency with corpus
def test_regression_matches_source_corpus():
    reg = _regression()
    gold = _load_jsonl(os.path.join(_CORPUS, "gold_qa.jsonl"))
    facts = _load_jsonl(os.path.join(_CORPUS, "versioned_facts.jsonl"))
    assert len(reg) == len(gold) + len(facts)
    # every source qa_id is present exactly once
    src_ids = {r["qa_id"] for r in gold} | {r["qa_id"] for r in facts}
    reg_ids = [r["qa_id"] for r in reg]
    assert set(reg_ids) == src_ids
    assert len(reg_ids) == len(set(reg_ids))  # no duplicates
    # benchmark fields present on every row
    for r in reg:
        assert r["document_family_id"] in {"finance_report", "finance_facts"}
        assert r["split"] in {"dev", "regression"}
        assert r["benchmark_set_id"] == "pdf_native_public_regression/v1"


def test_regression_is_deterministically_sorted():
    reg = _regression()
    ids = [r["qa_id"] for r in reg]
    assert ids == sorted(ids)


# --------------------------------------------------------------------------- leakage audit on real data
def test_real_fixture_leakage_audit_zero_overlap():
    from evaluation.pdf_native import assert_no_split_leakage
    reg = _regression()
    splits = {}
    for r in reg:
        splits.setdefault(r["split"], []).append(r)
    audit = assert_no_split_leakage(splits)
    assert audit["disjoint"] is True
    assert audit["intersection_size"] == 0
    assert audit["n_families"] == 2
    # families are disjoint across splits
    assert audit["families"]["dev"] == ["finance_report"]
    assert audit["families"]["regression"] == ["finance_facts"]


def test_manifest_matches_regression_and_audit():
    m = _manifest()
    reg = _regression()
    assert m["n_examples"] == len(reg)
    assert m["leakage_audit"]["disjoint"] is True
    assert m["leakage_audit"]["intersection_size"] == 0
    # category counts in the manifest match the raw set
    counts = {}
    for r in reg:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    assert m["category_counts"] == dict(sorted(counts.items()))


# --------------------------------------------------------------------------- claim check (spec gate)
def test_no_sealed_or_unseen_claim():
    """Spec: claiming `sealed`/`unseen` without a real protected label store must FAIL
    the claim check. We operate no protected store -> those words must not be *asserted*.
    They may appear only inside an explicit negation / `planned` note."""
    m = _manifest()
    assert m["role"] == "public_frozen_regression"
    assert m["protected_final"]["status"] == "planned"
    # the whole benchmark.yaml + manifest must not positively assert sealed/unseen
    bench_txt = open(os.path.join(_DIR, "benchmark.yaml"), encoding="utf-8").read().lower()
    for word in ("sealed", "unseen"):
        # allowed only in a negating context ("not 'sealed'", "removed ... unseen")
        for line in bench_txt.splitlines():
            if word in line:
                assert any(neg in line for neg in ("not ", "never", "remove", "without", "downgrad")), \
                    f"{word!r} asserted without negation: {line!r}"


def test_review_status_not_faked_human():
    """The agent must not fake human review; before owner review this stays pending."""
    m = _manifest()
    assert m["status"] == "owner_review_pending"
    assert m["n_human_reviewed"] == 0
    for r in _regression():
        assert r["review_status"] != "human_reviewed"


# --------------------------------------------------------------------------- benchmark.yaml contract
def test_benchmark_yaml_arms_all_measured_and_backed_by_raw():
    """Arms are now `measured` (real A100 run). Guard against *fabricated* results: every arm
    must be backed by committed raw per_example data, and summary.json must be a real
    (non-smoke) run whose aggregate regenerates from that raw data."""
    import glob, json
    from evaluation import pdf_native as PN
    with open(os.path.join(_DIR, "benchmark.yaml"), encoding="utf-8") as fh:
        b = yaml.safe_load(fh)
    arms = {a["id"]: a for a in b["experiment_arms"]}
    assert set(arms) == {"base_bf16", "sft_bf16", "sft_int4_ptq", "sft_int4_qat",
                         "base_bf16_retrieval", "sft_bf16_retrieval"}
    assert all(a["status"] == "measured" for a in arms.values())
    # retrieval baseline present (spec: not optional)
    assert "base_bf16_retrieval" in arms and "sft_bf16_retrieval" in arms
    # every arm is backed by committed raw per_example data (not fabricated numbers)
    v1 = os.path.join(_DIR, "historical_final", "v1")
    summary = json.load(open(os.path.join(v1, "summary.json"), encoding="utf-8"))
    assert summary["smoke"] is False and summary["base_model"] == "Qwen/Qwen3-8B"
    n_rows = len(_regression())
    for arm in arms:
        files = sorted(glob.glob(os.path.join(v1, "per_example", f"{arm}_seed*.jsonl")))
        assert files, f"arm {arm} has no committed raw per_example data"
        recs = [json.loads(l) for l in open(files[0], encoding="utf-8")]
        assert len(recs) == n_rows, f"{arm}: {len(recs)} records != {n_rows} eval rows"
        assert summary["arms"][arm]["status"] == "measured"
    # summary aggregate must regenerate from raw (honesty contract: never hand-typed)
    recs = [json.loads(l) for l in
            open(os.path.join(v1, "per_example", "sft_bf16_retrieval_seed42.jsonl"), encoding="utf-8")]
    assert abs(PN.aggregate(recs, k=4)["f1"] - 0.4442) < 0.02


def test_benchmark_yaml_metric_contract_complete():
    with open(os.path.join(_DIR, "benchmark.yaml"), encoding="utf-8") as fh:
        b = yaml.safe_load(fh)
    required = {"exact_match", "token_f1", "numeric_exact_rate", "citation_page_accuracy",
                "citation_span_accuracy", "retrieval_recall_at_k", "no_answer_retrieval_rate",
                "abstention_precision", "abstention_recall", "schema_validity_rate",
                "groundedness_rate", "pii_leakage_rate", "per_category_accuracy",
                "failure_taxonomy"}
    assert required.issubset(set(b["metrics"]))


def test_benchmark_yaml_ten_minimum_categories_present():
    with open(os.path.join(_DIR, "benchmark.yaml"), encoding="utf-8") as fh:
        b = yaml.safe_load(fh)
    cats = b["categories"]
    assert len(cats) == 10  # all 10 spec categories enumerated
    # each is either backed by the corpus (present/partial) or honestly marked planned
    for name, meta in cats.items():
        assert meta["status"] in {"present", "partial", "planned"}
    present = {n for n, m in cats.items() if m["status"] == "present"}
    assert {"single_evidence_fact", "table_or_chart_lookup", "numeric_currency_date_unit",
            "cross_page_evidence", "unanswerable_refusal", "prompt_injection_document"} <= present


def test_fairness_contract_present():
    with open(os.path.join(_DIR, "benchmark.yaml"), encoding="utf-8") as fh:
        b = yaml.safe_load(fh)
    fc = b["fairness_contract"]
    assert set(fc["shared_across_all_arms"]) >= {"current_final_set", "prompt_contract",
                                                 "tokenizer", "decode_parameters", "hardware_class"}
    assert set(fc["retrieval_arms_shared"]) >= {"parser", "corpus_version", "chunker", "top_k"}
    assert "3-seed" in fc["variance"] and "bootstrap" in fc["variance"]


# --------------------------------------------------------------------------- acceptance pre-registration
def test_acceptance_preregistered_thresholds_unchanged_after_eval():
    """After the run, evaluation_status flips to `evaluated` but the pre-registered THRESHOLDS
    must be byte-for-byte unchanged (moving a goalpost after seeing results is forbidden)."""
    with open(os.path.join(_DIR, "acceptance.yaml"), encoding="utf-8") as fh:
        acc = yaml.safe_load(fh)
    assert acc["pre_registered"] is True
    assert acc["frozen_before_run"] is True
    assert acc["evaluation_status"] == "evaluated"   # arms have been run
    # the frozen thresholds must be exactly the pre-registered values (not edited post-hoc)
    assert acc["quality_criteria"]["sft_improves_over_base"]["threshold"] == {"delta_gte": 0.03}
    assert acc["quality_criteria"]["abstention_on_unanswerable"]["threshold"] == {"gte": 0.90}
    assert acc["quality_criteria"]["citation_span_accuracy"]["threshold"] == {"gte": 0.80}
    assert acc["quality_criteria"]["groundedness"]["threshold"] == {"gte": 0.85}
    # honest verdicts recorded (including FAILs) — no fabricated all-pass
    verdicts = {k: v.get("status") for k, v in acc["quality_criteria"].items()}
    assert verdicts["sft_improves_over_base"] == "fail"       # closed-book SFT != fact injection
    assert verdicts["groundedness"] == "pass"
    # hard gates include leakage-zero-overlap and pii==0, and passed on real data
    hg = acc["hard_gates"]
    assert hg["leakage_zero_overlap"]["threshold"] == {"eq": 0}
    assert hg["pii_leakage"]["threshold"] == {"eq": 0.0}
    assert hg["pii_leakage"]["status"] == "pass"
    # retrieval necessity gate exists (spec: not optional)
    assert "mutable_fact_retrieval_beats_closed_book" in acc["retrieval_criteria"]
    # the practical closed-book fine-tuning effect is NOT claimed (honest)
    assert acc["retrieval_criteria"]["practical_effect_claim_allowed"]["status"] == "not_allowed"


# --------------------------------------------------------------------------- checksums integrity
def test_checksums_match_generated_files():
    with open(os.path.join(_DIR, "checksums.sha256"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split("  ", 1)
            path = os.path.join(_DIR, rel)
            assert os.path.exists(path), rel
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            assert h == digest, f"checksum mismatch for {rel}"


# --------------------------------------------------------------------------- rebuild determinism
def test_rebuild_is_deterministic(tmp_path):
    """Re-running the builder into a fresh dir reproduces the committed bytes exactly
    (aggregate/manifest are auto-generated from raw, nothing hand-typed)."""
    from workflows.pdf_native_post_training.benchmarks.pdf_native import build_benchmark as B
    out = str(tmp_path)
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)
    B.write_all(out_dir=out)
    for rel in ["public_regression.jsonl", "final_manifest.json",
                "splits/dev.json", "splits/regression.json", "checksums.sha256"]:
        new = open(os.path.join(out, rel), "rb").read()
        old = open(os.path.join(_DIR, rel), "rb").read()
        assert new == old, f"non-deterministic rebuild: {rel}"
