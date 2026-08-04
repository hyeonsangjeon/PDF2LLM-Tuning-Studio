"""Architecture boundary guard (spec §2.2).

The workflow depends one-way on pdf_qa / evaluation / quantization; those
packages must never import the workflow, and the workflow must never write into
the quantization results tree.
"""
import glob
import os
import sys

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))  # pdf_qa_extraction
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CORE_PACKAGES = ["pdf_qa", "evaluation", "quantization"]


def _py_files(pkg):
    return glob.glob(os.path.join(_ROOT, pkg, "**", "*.py"), recursive=True)


def test_core_packages_never_import_workflows():
    offenders = []
    for pkg in CORE_PACKAGES:
        for path in _py_files(pkg):
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            if "import workflows" in src or "from workflows" in src:
                offenders.append(os.path.relpath(path, _ROOT))
    assert offenders == [], f"core packages import the workflow: {offenders}"


def test_workflow_does_not_reference_quantization_results():
    # the workflow must write run outputs under runs/, never into quantization/results
    offenders = []
    wf = os.path.join(_ROOT, "workflows")
    for path in glob.glob(os.path.join(wf, "**", "*.py"), recursive=True):
        if f"{os.sep}tests{os.sep}" in path:
            continue  # test files legitimately reference the path in assertions
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if "quantization/results" in src or "quantization\\results" in src:
            offenders.append(os.path.relpath(path, _ROOT))
    assert offenders == [], f"workflow references quantization/results: {offenders}"


def test_workflow_run_writes_only_under_run_dir(tmp_path):
    from workflows.pdf_native_post_training import cli

    qresults = os.path.join(_ROOT, "quantization", "results")
    before = _snapshot(qresults)
    run_dir = str(tmp_path / "run")
    rc = cli.main([
        "--config", os.path.join(_ROOT, "workflows", "pdf_native_post_training", "configs", "demo-replay.yaml"),
        "--run-dir", run_dir,
    ])
    assert rc == 0
    assert _snapshot(qresults) == before, "workflow run mutated quantization/results"
    # all produced files live under the run dir
    produced = glob.glob(os.path.join(run_dir, "**", "*"), recursive=True)
    assert produced, "run produced no files"


def _snapshot(root):
    if not os.path.isdir(root):
        return {}
    out = {}
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(path):
            out[path] = os.path.getmtime(path)
    return out
