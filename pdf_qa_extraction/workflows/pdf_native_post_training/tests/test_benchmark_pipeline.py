"""P1-9 tests — pipeline benchmark metrics (schema-valid, honest not_measured)."""

from __future__ import annotations

import os

import pytest

from workflows.pdf_native_post_training import benchmark_pipeline as bp

_HERE = os.path.dirname(os.path.abspath(__file__))
_WF = os.path.dirname(_HERE)
_DEMO_CONFIG = os.path.join(_WF, "configs", "demo-replay.yaml")


@pytest.fixture(scope="module")
def demo_metrics(tmp_path_factory):
    run_dir = str(tmp_path_factory.mktemp("bench_run"))
    return bp.run_and_measure(_DEMO_CONFIG, run_dir)


def test_metrics_schema_valid(demo_metrics):
    assert bp.validate_metrics(demo_metrics) == []
    assert demo_metrics["schema_version"] == "pdf2llm-metrics/1"
    assert demo_metrics["kind"] == "pipeline"


def test_demo_expected_counts(demo_metrics):
    p = demo_metrics["pipeline"]
    assert p["n_documents"] == 1
    assert p["n_pages"] == 3
    assert p["elements"]["total"] == 35
    assert p["elements"]["text"] == 20
    assert p["elements"]["table"] == 15
    assert p["elements"]["figure"] == 0
    assert p["qa"] == {"raw": 26, "accepted": 26, "rejected": 0, "yield": 1.0}
    assert p["evidence_pass_rate"] == 1.0


def test_sources_present_with_hash(demo_metrics):
    roles = {s["role"]: s for s in demo_metrics["sources"]}
    assert "run_manifest" in roles and "config" in roles
    for s in demo_metrics["sources"]:
        assert isinstance(s["sha256"], str) and len(s["sha256"]) == 64


def test_not_measured_propagation(demo_metrics):
    """Unmeasurable values are explicit markers, never a fake 0."""
    p = demo_metrics["pipeline"]
    # CPU box: no GPU -> VRAM cannot be measured.
    assert p["peak_vram_mb"] == "not_measured"
    # No human recorded review time.
    assert p["manual_review_minutes"] == "not_measured"
    # Demo corpus has no figures -> linkage is not applicable, not 0.
    assert p["figure_caption_linkage_rate"] == "not_applicable"


def test_peak_ram_is_real_number(demo_metrics):
    p = demo_metrics["pipeline"]
    assert isinstance(p["peak_ram_mb"], (int, float)) and p["peak_ram_mb"] > 0
    assert isinstance(p["pages_per_sec"], (int, float)) and p["pages_per_sec"] > 0


def test_provider_usage_zero_on_replay(demo_metrics):
    # Recorded replay makes no external provider calls.
    assert demo_metrics["pipeline"]["provider_usage"]["calls"] == 0


def test_collect_is_reproducible_from_run_dir(tmp_path):
    """collect_metrics on an existing run dir yields identical pipeline metrics."""
    run_dir = str(tmp_path / "r")
    a = bp.run_and_measure(_DEMO_CONFIG, run_dir)
    b = bp.collect_metrics(run_dir, _DEMO_CONFIG)
    assert a["pipeline"]["elements"] == b["pipeline"]["elements"]
    assert a["pipeline"]["qa"] == b["pipeline"]["qa"]
    assert a["pipeline"]["evidence_pass_rate"] == b["pipeline"]["evidence_pass_rate"]
