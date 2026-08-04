"""Run-bundle contract tests (P0-3)."""
from __future__ import annotations

import os

from pdf_qa import run_bundle as rb


def _make(run_id, out_shas, ts="2026-01-01T00:00:00Z"):
    b = rb.RunBundle(run_id=run_id, command="pdf2llm run -c demo", generation_mode="recorded_replay")
    b.created_at_utc = ts
    b.code = {"git_sha": "abc", "git_dirty": False, "config_sha256": "cfg", "prompt_sha256": "p", "rubric_sha256": None}
    b.model = {"id": "m", "revision": "r"}
    b.dataset = {"id": "d", "revision": "1", "split": "dev", "example_id_hash": "eh"}
    b.seeds = [42, 43]
    b.inputs = [{"path": "in/a.pdf", "sha256": "aaa", "role": "pdf"}]
    b.outputs = [{"path": p, "sha256": s} for p, s in out_shas]
    b.environment = rb.environment_info()
    b.add_stage("parse", "completed")
    return b


def test_manifest_validates():
    m = _make("run-1", [("qa.jsonl", "111")]).to_manifest()
    assert rb.validate_manifest(m) == []


def test_fingerprint_stable_across_runid_time_and_output_path():
    a = _make("run-A", [("out/x.jsonl", "111")], ts="2026-01-01T00:00:00Z")
    b = _make("run-B", [("DIFFERENT/y.jsonl", "111")], ts="2030-12-31T23:59:59Z")
    # run_id, timestamp and output *paths* differ, but inputs/code/config/model are equal
    assert a.reproducibility_fingerprint() == b.reproducibility_fingerprint()


def test_fingerprint_changes_when_input_changes():
    a = _make("r", [("o", "1")])
    b = _make("r", [("o", "1")])
    b.inputs = [{"path": "in/a.pdf", "sha256": "DIFFERENT", "role": "pdf"}]
    assert a.reproducibility_fingerprint() != b.reproducibility_fingerprint()


def test_artifact_set_hash_tracks_outputs():
    a = _make("r", [("o.jsonl", "111")])
    b = _make("r", [("o.jsonl", "222")])
    assert a.artifact_set_hash() != b.artifact_set_hash()
    c = _make("r", [("o.jsonl", "111")])
    assert a.artifact_set_hash() == c.artifact_set_hash()


def test_validate_catches_missing_fields():
    m = _make("r", [("o", "1")]).to_manifest()
    del m["reproducibility_fingerprint"]
    assert any("reproducibility_fingerprint" in e for e in rb.validate_manifest(m))


def test_atomic_write_and_reload(tmp_path):
    b = _make("r", [("o", "1")])
    p = b.write(str(tmp_path))
    assert os.path.exists(p)
    import json

    reloaded = json.load(open(p, encoding="utf-8"))
    assert reloaded["run_id"] == "r"
    assert rb.validate_manifest(reloaded) == []


def test_no_tmp_file_left_behind(tmp_path):
    b = _make("r", [("o", "1")])
    b.write(str(tmp_path))
    leftovers = [f for f in os.listdir(tmp_path) if ".tmp." in f]
    assert leftovers == []
