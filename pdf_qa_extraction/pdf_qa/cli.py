"""``pdf2llm`` command-line entry point (composition root / launcher).

This launcher deliberately does NOT import the ``workflows`` package: the
one-way dependency rule (core never imports the workflow) is enforced by keeping
the workflow run behind a subprocess boundary. Core operations (``parse``,
``scan-secrets``) call ``pdf_qa`` directly.

Subcommands:
    pdf2llm run --config <cfg.yaml> [--run-dir DIR]
    pdf2llm demo-replay | demo-live-ollama | demo-train-smoke
    pdf2llm verify-demo                # run the replay demo and assert integrity
    pdf2llm build-fixture              # regenerate the synthetic demo fixture
    pdf2llm parse <file.pdf>           # parse a PDF into provenance elements
    pdf2llm scan-secrets [paths...]    # run the secret/PII scanner
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pdf_qa_extraction
_WF_CONFIGS = os.path.join(_PKG_ROOT, "workflows", "pdf_native_post_training", "configs")
_FIXTURE = os.path.join(_PKG_ROOT, "workflows", "pdf_native_post_training", "public_finance_demo")

_DEMO_CONFIGS = {
    "demo-replay": "demo-replay.yaml",
    "demo-live-ollama": "demo-live-ollama.yaml",
    "demo-train-smoke": "smoke-train.yaml",
}


def _subenv() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = _PKG_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_workflow(config_path: str, run_dir: Optional[str] = None, runs_root: Optional[str] = None) -> int:
    cmd = [sys.executable, "-m", "workflows.pdf_native_post_training.cli", "--config", config_path]
    if run_dir:
        cmd += ["--run-dir", run_dir]
    if runs_root:
        cmd += ["--runs-root", runs_root]
    return subprocess.call(cmd, cwd=_PKG_ROOT, env=_subenv())


def _cmd_run(args) -> int:
    cfg = args.config
    if not os.path.isabs(cfg) and not os.path.isfile(cfg):
        cand = os.path.join(_WF_CONFIGS, cfg)
        if os.path.isfile(cand):
            cfg = cand
    return _run_workflow(cfg, args.run_dir, args.runs_root)


def _cmd_demo(args, name: str) -> int:
    return _run_workflow(os.path.join(_WF_CONFIGS, _DEMO_CONFIGS[name]), args.run_dir, args.runs_root)


def _cmd_verify_demo(args) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "verify")
        rc = _run_workflow(os.path.join(_WF_CONFIGS, "demo-replay.yaml"), run_dir)
        if rc != 0:
            print("[verify-demo] FAIL: workflow returned non-zero", file=sys.stderr)
            return 1
        report = json.load(open(os.path.join(run_dir, "report.json")))
        checks = {
            "evidence_address_integrity==1.0": report["evidence_address_integrity"] == 1.0,
            "policy_quarantined==0": report["policy_quarantined"] == 0,
            "eval.em==1.0": report["eval"]["overall"]["em"] == 1.0,
            "eval.f1==1.0": report["eval"]["overall"]["f1"] == 1.0,
            "train_rows>0": report["train_rows_exported"] > 0,
        }
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if all(checks.values()):
            print("[verify-demo] PASS")
            return 0
        print("[verify-demo] FAIL", file=sys.stderr)
        return 1


def _cmd_build_fixture(args) -> int:
    return subprocess.call([sys.executable, os.path.join(_FIXTURE, "build_fixture.py")],
                           cwd=_FIXTURE, env=_subenv())


def _cmd_parse(args) -> int:
    from pdf_qa.provenance import parse_pdf

    doc = parse_pdf(args.pdf)
    print(json.dumps({"path": os.path.basename(args.pdf), "sha256": doc.sha256[:16],
                      "pages": doc.n_pages, "elements": len(doc.elements)}, indent=2))
    return 0


def _cmd_scan_secrets(args) -> int:
    scanner = os.path.join(_PKG_ROOT, "scripts", "scan_secrets.py")
    paths = args.paths or [_PKG_ROOT]
    return subprocess.call([sys.executable, scanner] + paths, cwd=_PKG_ROOT, env=_subenv())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdf2llm", description="PDF2LLM-Tuning-Studio launcher")
    sub = p.add_subparsers(dest="command", required=True)

    def _add_run_opts(sp):
        sp.add_argument("--run-dir", default=None)
        sp.add_argument("--runs-root", default="runs")

    r = sub.add_parser("run", help="run a workflow config")
    r.add_argument("--config", required=True)
    _add_run_opts(r)
    r.set_defaults(func=_cmd_run)

    for name in _DEMO_CONFIGS:
        d = sub.add_parser(name, help=f"run the {name} demo")
        _add_run_opts(d)
        d.set_defaults(func=lambda a, n=name: _cmd_demo(a, n))

    v = sub.add_parser("verify-demo", help="run demo-replay and assert integrity")
    v.set_defaults(func=_cmd_verify_demo)

    b = sub.add_parser("build-fixture", help="regenerate the synthetic demo fixture")
    b.set_defaults(func=_cmd_build_fixture)

    pa = sub.add_parser("parse", help="parse a PDF into provenance elements")
    pa.add_argument("pdf")
    pa.set_defaults(func=_cmd_parse)

    sc = sub.add_parser("scan-secrets", help="run the secret/PII scanner")
    sc.add_argument("paths", nargs="*")
    sc.set_defaults(func=_cmd_scan_secrets)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
