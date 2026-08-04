import json
import os
import sys

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from workflows.pdf_native_post_training import cli  # noqa: E402

_CONFIGS = os.path.join(_ROOT, "workflows", "pdf_native_post_training", "configs")


def _run(tmp_path, name="demo-replay"):
    run_dir = str(tmp_path / "run")
    rc = cli.main(["--config", os.path.join(_CONFIGS, f"{name}.yaml"), "--run-dir", run_dir])
    return rc, run_dir


def test_demo_replay_end_to_end(tmp_path):
    rc, run_dir = _run(tmp_path)
    assert rc == 0
    report = json.load(open(os.path.join(run_dir, "report.json")))
    assert report["evidence_address_integrity"] == 1.0
    assert report["policy_quarantined"] == 0
    assert report["train_rows_exported"] == 26
    assert report["eval"]["overall"]["em"] == 1.0
    assert report["eval"]["overall"]["f1"] == 1.0
    # every category present in eval
    assert set(report["eval"]["per_category"]) >= {
        "numeric_exact", "single_fact", "table_lookup", "cross_page",
        "prompt_injection", "unanswerable",
    }


def test_manifest_is_clean_and_valid(tmp_path):
    from pdf_qa.run_bundle import validate_manifest

    _, run_dir = _run(tmp_path)
    m = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    assert validate_manifest(m) == []
    # three distinct identifiers, no self-embedding
    assert m["run_id"] not in m["reproducibility_fingerprint"]
    assert m["reproducibility_fingerprint"] != m["artifact_set_hash"]
    # no absolute local paths / secrets leaked
    blob = json.dumps(m)
    for bad in ("/ai-work", "/home/", "/tmp/", "AKIA", "sk-", "BEGIN PRIVATE KEY"):
        assert bad not in blob, bad
    # outputs are workflow-relative, never under quantization/results
    for o in m["outputs"]:
        assert not o["path"].startswith("quantization/")


def test_fingerprint_is_path_independent(tmp_path):
    rd1 = str(tmp_path / "run_a")
    rd2 = str(tmp_path / "run_b")
    assert cli.main(["--config", os.path.join(_CONFIGS, "demo-replay.yaml"), "--run-dir", rd1]) == 0
    assert cli.main(["--config", os.path.join(_CONFIGS, "demo-replay.yaml"), "--run-dir", rd2]) == 0
    m1 = json.load(open(os.path.join(rd1, "run_manifest.json")))
    m2 = json.load(open(os.path.join(rd2, "run_manifest.json")))
    assert m1["reproducibility_fingerprint"] == m2["reproducibility_fingerprint"]
    assert m1["run_id"] != m2["run_id"]


def test_resume_skips_all_stages(tmp_path):
    _, run_dir = _run(tmp_path)
    # second run into the same dir: every stage should be skipped by hash
    rc = cli.main(["--config", os.path.join(_CONFIGS, "demo-replay.yaml"), "--run-dir", run_dir])
    assert rc == 0
    m = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    statuses = {s["name"]: s["status"] for s in m["stages"]}
    assert set(statuses.values()) == {"skipped"}
