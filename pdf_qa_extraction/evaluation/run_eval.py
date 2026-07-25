"""CLI entrypoint for the memoirist QA scorer.

Two modes (spec §5b):

* ``score``   — QC a single JSONL: write ``*_scored.jsonl`` + ``*_clean.jsonl``
  (strict-PASS, training-ready) + ``*_rejected.jsonl`` (with reasons) + a
  ``*_report.md`` (strict/lenient rates, per-dimension failures, run×chunk).
* ``compare`` — score several variants (persona/model runs) and emit one
  comparison table (+ per-variant reports). Replaces the manual v1-vs-v4 tally.

Examples::

    # QC one dataset with an LLM judge (Azure, judge model != generator model)
    python -m evaluation.run_eval score \
        --qa data/samples/qa_memoirist_v4_run1.jsonl \
        --source data/samples/memoir_sample_ko_long.txt \
        --judge-provider azure --judge-model gpt-5.4-mini --pairs-per-chunk 5

    # Deterministic Layer-1 only (no credentials / no judge)
    python -m evaluation.run_eval score --qa run1.jsonl --source src.txt --no-judge

    # Compare variants and reproduce the manual v1-vs-v4 aggregate
    python -m evaluation.run_eval compare --source src.txt \
        --runs data/samples/qa_memoirist_v4_run1.jsonl ... --name v4_vs_v1
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

# Allow ``python evaluation/run_eval.py`` as well as ``-m evaluation.run_eval``.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.qa_scorer import (  # noqa: E402
    Judge,
    LLMJudge,
    PairScore,
    RecordingJudge,
    ReplayJudge,
    Rubric,
    aggregate,
    aggregate_by_chunk,
    load_pairs,
    load_rubric,
    score_pairs,
    summarize_runs,
)

_DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
_REPORT_DIM_LABELS = {
    "format": "FORMAT",
    "register": "REGISTER(존댓말)",
    "first_person": "FIRST_PERSON",
    "grounded": "GROUNDED(날조)",
    "coherent": "COHERENT(비문/왜곡)",
    "voice_preserved": "VOICE_PRESERVED",
    "q_grounded": "Q_GROUNDED",
    "leading_q": "LEADING_Q(유도질문)",
}
_REPORT_DIM_ORDER = (
    "format",
    "register",
    "first_person",
    "grounded",
    "coherent",
    "voice_preserved",
    "q_grounded",
    "leading_q",
)


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------
def load_source(path: str, strategy: str = "fast") -> str:
    """Return the grounding text for ``path``.

    ``.txt`` is read directly; a PDF is partitioned with the existing ``pdf_qa``
    parser (imported lazily so the light Layer-1 path never needs it).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    if ext == ".pdf":
        from pdf_qa.extract import extract_document_layout

        layout = extract_document_layout(path, strategy=strategy, gpu_boost=False)
        chunks = [c.text.strip() for c in layout.text_chunks if (c.text or "").strip()]
        return "\n\n".join(chunks).strip()
    raise ValueError(f"Unsupported --source type '{ext}' (use .pdf, .txt or .md).")


# ---------------------------------------------------------------------------
# Judge construction
# ---------------------------------------------------------------------------
def build_judge(args, rubric: Rubric) -> Optional[Judge]:
    """Build the judge selected on the CLI (or ``None`` for Layer-1 only)."""
    if getattr(args, "no_judge", False):
        return None
    if getattr(args, "replay_cache", None):
        return ReplayJudge.from_file(args.replay_cache)
    judge = LLMJudge(
        prompt_template=rubric.judge_prompt,
        provider=args.judge_provider,
        model=args.judge_model,
        temperature=args.judge_temperature,
        api_version=args.judge_api_version,
    )
    print(
        f"[judge] provider={judge.provider} model={getattr(judge, 'model', '?')} "
        f"temperature={judge.temperature} (독립 호출)"
    )
    if getattr(args, "record_judge", None):
        return RecordingJudge(judge)
    return judge


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _pct(numerator: int, denominator: int) -> str:
    return f"{(100.0 * numerator / denominator):.0f}%" if denominator else "—"


def render_single_report(name: str, scores: Sequence[PairScore], agg: dict) -> str:
    total = agg["total"]
    judged = agg["judged"]
    lines = [
        f"# QA 스코어러 리포트 — `{name}`",
        "",
        f"- 총 쌍: **{total}** (judge 판정: {judged})",
        f"- **strict PASS**: {agg['strict_pass']}/{total} "
        f"({_pct(agg['strict_pass'], total)})  "
        f"— REGISTER∧FIRST_PERSON∧GROUNDED∧COHERENT∧VOICE_PRESERVED∧Q_GROUNDED",
        f"- **lenient PASS**: {agg['lenient_pass']}/{total} "
        f"({_pct(agg['lenient_pass'], total)})  — GROUNDED∧COHERENT∧REGISTER",
        "",
        "## 차원별 실패 수",
        "",
        "| 차원 | 실패 | 경고 |",
        "|---|---|---|",
    ]
    for dim in _REPORT_DIM_ORDER:
        label = _REPORT_DIM_LABELS[dim]
        lines.append(f"| {label} | {agg['dim_fail'].get(dim, 0)} | {agg['dim_warn'].get(dim, 0)} |")

    by_chunk = aggregate_by_chunk(scores)
    if by_chunk:
        lines += ["", "## 청크 축 집계 (존댓말 lock 확인용)", "",
                  "| chunk | 쌍 | strict | REGISTER 실패 |", "|---|---|---|---|"]
        for chunk, cagg in by_chunk.items():
            lines.append(
                f"| {chunk} | {cagg['total']} | "
                f"{cagg['strict_pass']}/{cagg['total']} | "
                f"{cagg['dim_fail'].get('register', 0)} |"
            )

    lines += ["", "## 쌍별 판정", "", "| # | strict | 실패 차원 | 답변(앞 40자) |", "|---|---|---|---|"]
    for s in scores:
        failed = ", ".join(s.failed_dims()) or "—"
        mark = "✅" if s.strict_pass else "❌"
        preview = s.answer[:40].replace("|", "／").replace("\n", " ")
        lines.append(f"| {s.index} | {mark} | {failed} | {preview} |")
    return "\n".join(lines) + "\n"


def render_compare_report(name: str, variants: Sequence[dict]) -> str:
    """``variants``: list of {name, runs:[{name, agg}], summary aggregates}."""
    lines = [
        f"# QA 스코어러 비교 리포트 — `{name}`",
        "",
        "변이별 재판정 결과(스코어러). 수기 v1-vs-v4 집계를 대체.",
        "",
        "| 변이 | 런 | 총쌍 | strict PASS | lenient PASS | REGISTER실패 | GROUNDED실패 | COHERENT실패 | VOICE실패 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in variants:
        strict = summarize_runs([r["agg"] for r in v["runs"]], "strict_pass")
        lenient = summarize_runs([r["agg"] for r in v["runs"]], "lenient_pass")
        total = sum(r["agg"]["total"] for r in v["runs"])
        n_runs = len(v["runs"])
        reg = sum(r["agg"]["dim_fail"].get("register", 0) for r in v["runs"])
        gr = sum(r["agg"]["dim_fail"].get("grounded", 0) for r in v["runs"])
        co = sum(r["agg"]["dim_fail"].get("coherent", 0) for r in v["runs"])
        vo = sum(r["agg"]["dim_fail"].get("voice_preserved", 0) for r in v["runs"])
        strict_sum = sum(r["agg"]["strict_pass"] for r in v["runs"])
        lenient_sum = sum(r["agg"]["lenient_pass"] for r in v["runs"])
        lines.append(
            f"| **{v['name']}** | {n_runs} | {total} | "
            f"{strict_sum}/{total} ({_pct(strict_sum, total)}); "
            f"run min/max/mean {strict['min']}/{strict['max']}/{strict['mean']:.1f} | "
            f"{lenient_sum}/{total} ({_pct(lenient_sum, total)}) | "
            f"{reg} | {gr} | {co} | {vo} |"
        )
    lines += ["", "## 런별 상세", ""]
    for v in variants:
        lines.append(f"### {v['name']}")
        lines.append("")
        lines.append("| 런 | 총쌍 | strict | lenient | REG실패 | GROUND실패 | COHER실패 | VOICE실패 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in v["runs"]:
            a = r["agg"]
            lines.append(
                f"| {r['name']} | {a['total']} | {a['strict_pass']} | {a['lenient_pass']} | "
                f"{a['dim_fail'].get('register',0)} | {a['dim_fail'].get('grounded',0)} | "
                f"{a['dim_fail'].get('coherent',0)} | {a['dim_fail'].get('voice_preserved',0)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _default_name(qa_path: str) -> str:
    return os.path.splitext(os.path.basename(qa_path))[0]


def run_score(args) -> int:
    rubric = load_rubric(args.rubric)
    source = load_source(args.source, strategy=args.strategy)
    judge = build_judge(args, rubric)
    pairs = load_pairs(args.qa)
    name = args.name or _default_name(args.qa)

    scores = score_pairs(
        pairs, source, rubric, judge=judge, pairs_per_chunk=args.pairs_per_chunk
    )
    agg = aggregate(scores)

    os.makedirs(args.out_dir, exist_ok=True)
    from evaluation.qa_scorer import write_clean_and_rejected, write_scored

    scored_path = os.path.join(args.out_dir, f"{name}_scored.jsonl")
    clean_path = os.path.join(args.out_dir, f"{name}_clean.jsonl")
    rejected_path = os.path.join(args.out_dir, f"{name}_rejected.jsonl")
    report_path = os.path.join(args.out_dir, f"{name}_report.md")

    write_scored(scores, scored_path)
    kept, dropped = write_clean_and_rejected(scores, clean_path, rejected_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_single_report(name, scores, agg))

    if isinstance(judge, RecordingJudge) and args.record_judge:
        judge.dump(args.record_judge)
        print(f"[judge] recorded {len(judge.records)} verdict(s) -> {args.record_judge}")

    print(
        f"[score] {name}: {agg['total']}쌍 → strict {agg['strict_pass']} "
        f"({_pct(agg['strict_pass'], agg['total'])}), "
        f"lenient {agg['lenient_pass']} ({_pct(agg['lenient_pass'], agg['total'])}); "
        f"clean {kept} / rejected {dropped}"
    )
    print(f"[score] outputs: {scored_path} | {clean_path} | {rejected_path} | {report_path}")
    return 0


def _variant_from_runs(
    variant_name: str,
    run_paths: Sequence[str],
    source: str,
    rubric: Rubric,
    judge: Optional[Judge],
    pairs_per_chunk: Optional[int],
) -> dict:
    runs = []
    for path in run_paths:
        pairs = load_pairs(path)
        scores = score_pairs(pairs, source, rubric, judge=judge, pairs_per_chunk=pairs_per_chunk)
        runs.append({"name": _default_name(path), "agg": aggregate(scores), "scores": scores})
    return {"name": variant_name, "runs": runs}


def run_compare(args) -> int:
    rubric = load_rubric(args.rubric)
    source = load_source(args.source, strategy=args.strategy)
    judge = build_judge(args, rubric)
    name = args.name or "compare"

    variants: List[dict] = []
    if args.variant:
        for spec in args.variant:
            if "=" not in spec:
                raise ValueError(f"--variant expects NAME=path1,path2 (got {spec!r})")
            vname, paths = spec.split("=", 1)
            run_paths = [p for p in paths.split(",") if p]
            variants.append(
                _variant_from_runs(vname, run_paths, source, rubric, judge, args.pairs_per_chunk)
            )
    else:
        variants.append(
            _variant_from_runs("runs", list(args.runs), source, rubric, judge, args.pairs_per_chunk)
        )

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, f"{name}_compare_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_compare_report(name, variants))

    if isinstance(judge, RecordingJudge) and args.record_judge:
        judge.dump(args.record_judge)
        print(f"[judge] recorded {len(judge.records)} verdict(s) -> {args.record_judge}")

    for v in variants:
        strict_sum = sum(r["agg"]["strict_pass"] for r in v["runs"])
        total = sum(r["agg"]["total"] for r in v["runs"])
        print(f"[compare] {v['name']}: strict {strict_sum}/{total} ({_pct(strict_sum, total)})")
    print(f"[compare] report: {report_path}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _add_common_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="Grounding source (.pdf | .txt | .md).")
    parser.add_argument("--strategy", default="fast", help="unstructured strategy for a PDF source.")
    parser.add_argument("--rubric", default=None, help="Override rubric.yaml path.")
    parser.add_argument("--judge-provider", default="azure", help="Judge provider (azure | openai).")
    parser.add_argument("--judge-model", default=None, help="Judge deployment/model (≠ generator ideal).")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature (spec: 0).")
    parser.add_argument("--judge-api-version", default=None, help="Azure OpenAI API version override.")
    parser.add_argument("--no-judge", action="store_true", help="Layer-1 only (deterministic, no LLM).")
    parser.add_argument("--replay-cache", default=None, help="Replay judge verdicts from a cache JSON.")
    parser.add_argument("--record-judge", default=None, help="Write judge verdicts to a cache JSON.")
    parser.add_argument("--pairs-per-chunk", type=int, default=None, help="Enable run×chunk aggregation.")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="Output directory.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memoirist QA evaluation scorer.")
    sub = parser.add_subparsers(dest="mode", required=True)

    score = sub.add_parser("score", help="QC a single JSONL dataset.")
    score.add_argument("--qa", required=True, help="JSONL to score.")
    score.add_argument("--name", default=None, help="Output basename (default: qa filename).")
    _add_common_judge_args(score)
    score.set_defaults(func=run_score)

    compare = sub.add_parser("compare", help="Compare several variants / runs.")
    compare.add_argument("--runs", nargs="+", default=[], help="JSONL runs to score together.")
    compare.add_argument(
        "--variant", action="append", default=[],
        help="Named variant as NAME=path1,path2 (repeatable). Overrides --runs.",
    )
    compare.add_argument("--name", default=None, help="Output basename.")
    _add_common_judge_args(compare)
    compare.set_defaults(func=run_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
