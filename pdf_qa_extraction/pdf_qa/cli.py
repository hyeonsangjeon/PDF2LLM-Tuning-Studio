"""``pdf2llm`` command-line entry point (composition root / launcher).

This launcher deliberately does NOT import the ``workflows`` package: the
one-way dependency rule (core never imports the workflow) is enforced by keeping
the workflow run behind a subprocess boundary. Core operations (``parse``,
``scan-secrets``) call ``pdf_qa`` directly.

Subcommands:
    pdf2llm run --config <cfg.yaml> [--run-dir DIR]
    pdf2llm demo-replay | demo-live-ollama | demo-train-smoke
    pdf2llm verify-demo                # run the replay demo and assert integrity
    pdf2llm ask [-q "질문"]            # replay the real A100 answers for a question
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


def _cmd_ask(args) -> int:
    # Delegated to the workflow via subprocess (core never imports workflows).
    cmd = [sys.executable, "-m", "workflows.pdf_native_post_training.ask_demo"]
    if args.question:
        cmd += ["--question", args.question]
    if args.qa_id:
        cmd += ["--qa-id", args.qa_id]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.list:
        cmd += ["--list"]
    if getattr(args, "hf", None):
        cmd += ["--hf", args.hf]
    if getattr(args, "no_retrieval", False):
        cmd += ["--no-retrieval"]
    if getattr(args, "max_new_tokens", None) is not None:
        cmd += ["--max-new-tokens", str(args.max_new_tokens)]
    if args.live:
        cmd += ["--live"]
    if args.model:
        cmd += ["--model", args.model]
    return subprocess.call(cmd, cwd=_PKG_ROOT, env=_subenv())


def _cmd_cook_demo(args) -> int:
    # Delegated to the workflow via subprocess (core never imports workflows/torch).
    cmd = [sys.executable, "-m", "workflows.pdf_native_post_training.cook_demo",
           "--model", args.model, "--out", args.out]
    if args.train:
        cmd += ["--train", args.train]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.max_seq_len is not None:
        cmd += ["--max-seq-len", str(args.max_seq_len)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.device:
        cmd += ["--device", args.device]
    if getattr(args, "dry_run", False):
        cmd += ["--dry-run"]
    return subprocess.call(cmd, cwd=_PKG_ROOT, env=_subenv())


def _cmd_publish_hf(args) -> int:
    # Delegated to the workflow via subprocess (core never imports workflows).
    cmd = [sys.executable, "-m", "workflows.pdf_native_post_training.publish_hf",
           "--model-dir", args.model_dir, "--repo-id", args.repo_id]
    if args.arm:
        cmd += ["--arm", args.arm]
    if args.base_model:
        cmd += ["--base-model", args.base_model]
    if args.summary:
        cmd += ["--summary", args.summary]
    if args.token:
        cmd += ["--token", args.token]
    if args.private:
        cmd += ["--private"]
    if getattr(args, "reference_scores", False):
        cmd += ["--reference-scores"]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.no_card:
        cmd += ["--no-card"]
    return subprocess.call(cmd, cwd=_PKG_ROOT, env=_subenv())


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

    ak = sub.add_parser("ask", help="학습된 모델에 질문 → 답이 짠! (실제 A100 결과 오프라인 재생)")
    ak.add_argument("-q", "--question", default=None, help="질문(부분 문자열) 또는 qa_id")
    ak.add_argument("--qa-id", dest="qa_id", default=None, help="정확한 qa_id (예: q000)")
    ak.add_argument("--seed", type=int, default=42, help="표시할 seed (기본 42)")
    ak.add_argument("--list", action="store_true", help="사용 가능한 질문 목록")
    ak.add_argument("--hf", metavar="REPO_OR_DIR", default=None,
                    help="파인튜닝 가중치를 실제 로드해 실시간 추론 (HF repo id 또는 로컬 경로)")
    ak.add_argument("--no-retrieval", dest="no_retrieval", action="store_true",
                    help="--hf 시 검색을 끄고 closed-book 추론")
    ak.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=64,
                    help="--hf 생성 토큰 상한 (기본 64)")
    ak.add_argument("--live", action="store_true", help="로컬 Ollama로 임의 문장 실시간 답변")
    ak.add_argument("--model", default="qwen2.5:7b-instruct", help="--live 시 Ollama 모델 태그")
    ak.set_defaults(func=_cmd_ask)

    ck = sub.add_parser("cook-demo",
                        help="GPU 없이 소형 chat 모델을 실제 SFT 파인튜닝 → ask --hf 로 로드")
    ck.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="베이스 chat 모델 (기본 Qwen/Qwen2.5-0.5B-Instruct)")
    ck.add_argument("--out", default="runs/cook_demo", help="출력 디렉터리 (기본 runs/cook_demo)")
    ck.add_argument("--train", default=None, help="학습 코퍼스 train.jsonl 경로")
    ck.add_argument("--max-steps", dest="max_steps", type=int, default=None, help="SFT 스텝 수")
    ck.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=None, help="토큰 상한")
    ck.add_argument("--lr", type=float, default=None, help="학습률")
    ck.add_argument("--limit", type=int, default=None, help="사용할 학습 예시 수")
    ck.add_argument("--device", default=None, help="cpu / cuda (기본: 자동)")
    ck.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="학습 없이 변환된 학습 행 수만 출력(토치 불필요)")
    ck.set_defaults(func=_cmd_cook_demo)

    ph = sub.add_parser("publish-hf",
                        help="파인튜닝 가중치를 HuggingFace Hub에 업로드 (모델 카드 자동 생성)")
    ph.add_argument("--model-dir", dest="model_dir", required=True,
                    help="업로드할 학습 산출물 디렉터리 (예: artifacts/sft_bf16_seed42)")
    ph.add_argument("--repo-id", dest="repo_id", required=True,
                    help="대상 HF 저장소 (예: your-name/pdf2llm-sft-qwen3-8b)")
    ph.add_argument("--arm", default=None, help="카드 점수 기준 arm (기본 sft_bf16_retrieval)")
    ph.add_argument("--base-model", dest="base_model", default=None,
                    help="베이스 모델 (기본: summary.json 값)")
    ph.add_argument("--summary", default=None, help="점수 출처 summary.json 경로")
    ph.add_argument("--token", default=None, help="HF write 토큰 (기본: $HF_TOKEN)")
    ph.add_argument("--private", action="store_true", help="비공개 저장소로 생성")
    ph.add_argument("--reference-scores", dest="reference_scores", action="store_true",
                    help="점수표를 '8B 벤치마크 참조값'으로 라벨 (소형 데모 업로드용; "
                         "베이스가 8B와 다르면 자동)")
    ph.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="업로드 없이 카드/파일 목록만 출력(토큰 불필요)")
    ph.add_argument("--no-card", dest="no_card", action="store_true",
                    help="README.md 모델 카드 작성/포함 생략")
    ph.set_defaults(func=_cmd_publish_hf)

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
