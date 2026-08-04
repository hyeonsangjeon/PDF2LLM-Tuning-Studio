"""Tests for the benchmark aggregation report (spec P1-2).

Verifies the committed derived tables (vllm_throughput.json, three_way_table.json)
reproduce exactly from the read-only raw JSON, that --emit never mutates results/,
that a corrupted raw input is caught, and that the derived calcs (crossover, ratios,
VRAM saving, mean/std) are correct. Import-light (PyYAML + stdlib; no torch).
"""
import json
import os
import sys

import pytest

_QDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # quantization/
if _QDIR not in sys.path:
    sys.path.insert(0, _QDIR)

import v2_report as R  # noqa: E402


# ------------------------------------------------------------------- reproduction
def test_check_historical_passes_on_committed_tables():
    rc = R._check_historical(R.DEFAULT_RESULTS, R.DEFAULT_CONFIG)
    assert rc == 0


def test_throughput_regenerates_exactly():
    doc, inputs = R.build_throughput(R.DEFAULT_RESULTS)
    hist = R.load_json(os.path.join(R.DEFAULT_RESULTS, "vllm_throughput.json"))
    assert R.hash_obj(doc) == R.hash_obj(hist), R.diff_pointers(doc, hist)
    assert len(inputs) == 5


def test_three_way_regenerates_exactly():
    doc, inputs = R.build_three_way(R.DEFAULT_RESULTS, R.DEFAULT_CONFIG)
    hist = R.load_json(os.path.join(R.DEFAULT_RESULTS, "three_way_table.json"))
    assert R.hash_obj(doc) == R.hash_obj(hist), R.diff_pointers(doc, hist)
    assert len(inputs) == 9  # 3 methods x 3 seeds


# ------------------------------------------------------------------- derived calcs
def test_crossover_batch():
    bf16 = [{"batch": 1, "throughput_tok_s": 87.6}, {"batch": 4, "throughput_tok_s": 342.7},
            {"batch": 16, "throughput_tok_s": 1031.5}]
    int4 = [{"batch": 1, "throughput_tok_s": 124.4}, {"batch": 4, "throughput_tok_s": 508.0},
            {"batch": 16, "throughput_tok_s": 327.3}]
    assert R.crossover_batch(bf16, int4) == 16          # bf16 first overtakes at 16
    assert R.crossover_batch(int4, bf16) == 1           # (swapped) int4 already ahead at 1


def test_crossover_none_when_never_overtakes():
    bf16 = [{"batch": 1, "throughput_tok_s": 10.0}, {"batch": 4, "throughput_tok_s": 20.0}]
    int4 = [{"batch": 1, "throughput_tok_s": 100.0}, {"batch": 4, "throughput_tok_s": 200.0}]
    assert R.crossover_batch(bf16, int4) is None


def test_vram_saving_x():
    assert R.vram_saving_x(15.27, 6.05) == 2.5
    assert R.vram_saving_x(15.27, 6.05, ndigits=3) == 2.524
    assert R.vram_saving_x(10.0, 0) is None


def test_single_stream_ratios_match_committed_derived():
    doc, _ = R.build_throughput(R.DEFAULT_RESULTS)
    d = doc["derived"]
    assert d["single_stream_throughput_int4_vs_bf16"] == 1.42     # 124.4 / 87.6
    assert d["single_stream_e2e_int4_faster_x"] == 1.896          # 1.4691 / 0.775
    assert d["single_stream_ttft_bf16_faster_x"] == 9.205         # 0.2964 / 0.0322
    assert d["batched_bf16_vs_int4_at_max_batch"] == 7.767        # 3959.0 / 509.7
    assert d["crossover_batch_bf16_overtakes"] == 16


def test_mean_std_population_ddof0():
    # f1 across seeds 42/43/44 for A_bf16 -> committed mean 94.83, std 0.187
    ms = R.mean_std([95.034, 94.874, 94.583])
    assert ms == {"mean": 94.83, "std": 0.187, "n": 3}
    assert R.mean_std([])["n"] == 0
    assert R.mean_std([5.0])["std"] == 0.0


def test_mean_std_matches_v2_pipeline_if_available():
    """Pin our local aggregation to the pipeline's, without a hard torch dependency."""
    pytest.importorskip("torch")
    from quantization import v2_pipeline
    rows = R.load_json(os.path.join(R.DEFAULT_RESULTS, "three_way_table.json"))["per_seed"]
    assert R.aggregate_seeds(rows) == v2_pipeline.aggregate_seeds(rows)


# ------------------------------------------------------------- emit / non-mutation
def test_emit_writes_runs_and_does_not_touch_results(tmp_path):
    before = {n: R.sha256_file(os.path.join(R.DEFAULT_RESULTS, n))
              for n in ("vllm_throughput.json", "three_way_table.json")}
    rc = R._emit(R.DEFAULT_RESULTS, R.DEFAULT_CONFIG, str(tmp_path), "utest", ["--emit"])
    assert rc == 0
    rep = tmp_path / "utest" / "quantization" / "report"
    assert (rep / "vllm_throughput.json").exists()
    assert (rep / "three_way_table.json").exists()
    prov = json.loads((rep / "provenance.json").read_text())
    assert prov["argv"] == ["--emit"]
    assert set(prov["reports"]) == {"vllm_throughput.json", "three_way_table.json"}
    assert all("sha256" in i for i in prov["reports"]["vllm_throughput.json"]["inputs"])
    after = {n: R.sha256_file(os.path.join(R.DEFAULT_RESULTS, n))
             for n in ("vllm_throughput.json", "three_way_table.json")}
    assert before == after, "results/ must remain read-only historical input"


def test_emitted_throughput_matches_historical(tmp_path):
    R._emit(R.DEFAULT_RESULTS, R.DEFAULT_CONFIG, str(tmp_path), "utest2", ["--emit"])
    emitted = R.load_json(str(tmp_path / "utest2" / "quantization" / "report" / "vllm_throughput.json"))
    hist = R.load_json(os.path.join(R.DEFAULT_RESULTS, "vllm_throughput.json"))
    assert R.hash_obj(emitted) == R.hash_obj(hist)


# ---------------------------------------------------------------- corruption caught
def test_corrupted_raw_input_is_detected(tmp_path):
    # Copy raw inputs, perturb one sweep number, expect check-historical to FAIL.
    import shutil
    rd = tmp_path / "results"
    rd.mkdir()
    for fn in os.listdir(R.DEFAULT_RESULTS):
        if fn.endswith(".json"):
            shutil.copy(os.path.join(R.DEFAULT_RESULTS, fn), rd / fn)
    bench = json.loads((rd / "bench_A.json").read_text())
    bench["sweep"][0]["throughput_tok_s"] = 999.9        # corrupt batch-1 bf16 throughput
    (rd / "bench_A.json").write_text(json.dumps(bench))
    rc = R._check_historical(str(rd), R.DEFAULT_CONFIG)
    assert rc == 1


def test_diff_pointers_reports_changes():
    a = {"x": {"y": 1}, "l": [1, 2]}
    b = {"x": {"y": 2}, "l": [1, 3]}
    diffs = R.diff_pointers(a, b)
    assert any("/x/y" in d for d in diffs)
    assert any("/l/1" in d for d in diffs)
