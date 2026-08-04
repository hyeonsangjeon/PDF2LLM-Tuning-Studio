import json
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pdf_qa.run_bundle import RunBundle  # noqa: E402
from workflows.pdf_native_post_training.stages.harness import (  # noqa: E402
    Stage, StageContext, StageError, run_pipeline,
)


def _ctx(tmp_path):
    return StageContext(config={}, run_dir=str(tmp_path / "run"), base_dir=str(tmp_path),
                        outputs={}, bundle=RunBundle(run_id="t", command="test"))


def _stage(name, value, counter):
    def run(ctx):
        counter[name] = counter.get(name, 0) + 1
        return {"value": value}
    return Stage(name, lambda ctx: {"name": name, "value": value}, run)


def test_partial_failure_raises_and_records_failed(tmp_path):
    ctx = _ctx(tmp_path)

    def boom(ctx):
        raise ValueError("kaboom")

    stages = [_stage("a", 1, {}), Stage("b", lambda c: {"n": "b"}, boom), _stage("c", 3, {})]
    with pytest.raises(StageError) as ei:
        run_pipeline(stages, ctx)
    assert ei.value.stage == "b"
    m = json.load(open(os.path.join(ctx.run_dir, "run_manifest.json")))
    statuses = {s["name"]: s["status"] for s in m["stages"]}
    assert statuses["a"] == "completed"
    assert statuses["b"] == "failed"
    assert "c" not in statuses  # pipeline stopped at failure (partial == fail)
    assert m["status"] == "failed"


def test_resume_by_hash_skips_recompute(tmp_path):
    counter = {}
    stages = [_stage("a", 1, counter), _stage("b", 2, counter)]

    ctx1 = _ctx(tmp_path)
    run_pipeline(stages, ctx1)
    assert counter == {"a": 1, "b": 1}

    # rerun into same dir -> run() must NOT be called again
    ctx2 = StageContext(config={}, run_dir=ctx1.run_dir, base_dir=str(tmp_path),
                        outputs={}, bundle=RunBundle(run_id="t2", command="test"))
    summary = run_pipeline(stages, ctx2)
    assert counter == {"a": 1, "b": 1}  # unchanged
    assert {s["status"] for s in summary["stages"]} == {"skipped"}


def test_signature_change_forces_recompute(tmp_path):
    counter = {}
    ctx1 = _ctx(tmp_path)
    run_pipeline([_stage("a", 1, counter)], ctx1)
    assert counter["a"] == 1
    # different signature value -> must recompute
    ctx2 = StageContext(config={}, run_dir=ctx1.run_dir, base_dir=str(tmp_path),
                        outputs={}, bundle=RunBundle(run_id="t3", command="test"))
    run_pipeline([_stage("a", 999, counter)], ctx2)
    assert counter["a"] == 2
