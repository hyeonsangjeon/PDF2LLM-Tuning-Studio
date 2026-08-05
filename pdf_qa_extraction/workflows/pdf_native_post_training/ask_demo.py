"""``pdf2llm ask`` — 학습된 모델에 질문하면 답이 "짠!" 하고 나오는 데모.

클론한 사람이 **아무 설정 없이** 한 줄로 "질문 → 답" 경험을 얻도록 만든 진입점입니다.
두 가지 경로가 있습니다.

1. **기본(오프라인·무설정)** — 실제 Azure A100에서 정주행한 6개 모델(Base/SFT/PTQ/QAT ± 검색)의
   답을 그대로 **재생(replay)** 해, 같은 질문을 *closed-book* vs *retrieval* 로 나란히 보여줍니다.
   GPU·API 키·네트워크가 전혀 필요 없습니다(커밋된 raw 예측을 읽을 뿐입니다).

2. ``--live`` — 로컬 Ollama로 **임의의 문장**을 실시간 답변(합성 PDF를 근거로). Ollama 데몬만 있으면
   되고 클라우드·GPU는 필요 없습니다. CI에서는 실행하지 않는 선택 경로입니다.

데이터 출처(1번 경로):
``benchmarks/pdf_native/historical_final/v1/per_example/<arm>_seed<seed>.jsonl``
— 각 행에 실제 질문(``_question``)·정답(``_gold``)·모델 예측(``_pred``)과 EM/F1·기권 여부가 들어 있습니다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

_DIR = os.path.dirname(os.path.abspath(__file__))
_PER_EXAMPLE = os.path.join(
    _DIR, "benchmarks", "pdf_native", "historical_final", "v1", "per_example"
)
_DOCS = os.path.join(_DIR, "public_finance_demo", "docs")

# 표시 순서 + 사람이 읽는 라벨. (arm 파일명, 라벨)
_CLOSED_BOOK: List[Tuple[str, str]] = [
    ("base_bf16", "Base (파인튜닝 전)"),
    ("sft_bf16", "SFT (파인튜닝)"),
    ("sft_int4_ptq", "SFT+INT4 PTQ (압축)"),
    ("sft_int4_qat", "SFT+INT4 QAT (압축)"),
]
_RETRIEVAL: List[Tuple[str, str]] = [
    ("base_bf16_retrieval", "Base + 검색"),
    ("sft_bf16_retrieval", "SFT + 검색"),
]
_BEST_ARM = "sft_bf16_retrieval"  # 벤치마크 최고 성능 arm
_DEFAULT_QID = "q000"  # retrieval 효과가 가장 잘 드러나는 기본 질문
_ALL_ARMS = _CLOSED_BOOK + _RETRIEVAL


# ---------------------------------------------------------------- 데이터 로딩

def _arm_path(arm: str, seed: int) -> Optional[str]:
    """arm+seed 파일 경로. 없으면 seed 42(측정된 기준 seed)로 폴백."""
    p = os.path.join(_PER_EXAMPLE, f"{arm}_seed{seed}.jsonl")
    if os.path.isfile(p):
        return p
    p42 = os.path.join(_PER_EXAMPLE, f"{arm}_seed42.jsonl")
    return p42 if os.path.isfile(p42) else None


def _load_arm(arm: str, seed: int) -> Dict[str, dict]:
    path = _arm_path(arm, seed)
    if path is None:
        return {}
    rows: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["qa_id"]] = r
    return rows


def _load_all(seed: int) -> Dict[str, Dict[str, dict]]:
    return {arm: _load_arm(arm, seed) for arm, _ in _ALL_ARMS}


def _question_catalog(seed: int) -> List[dict]:
    """질문 목록(가장 완전한 retrieval arm 기준)."""
    rows = _load_arm("sft_bf16_retrieval", seed) or _load_arm("sft_bf16", seed)
    out = [
        {
            "qa_id": r["qa_id"],
            "question": r.get("_question", ""),
            "category": r.get("category", ""),
            "answerable": r.get("answerable"),
            "gold": r.get("_gold"),
        }
        for r in rows.values()
    ]
    out.sort(key=lambda d: d["qa_id"])
    return out


# ---------------------------------------------------------------- 표시 헬퍼

def _disp_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_width(s))


def _clip(s: Optional[str], n: int = 52) -> str:
    s = (s or "").replace("\n", " ").strip()
    if _disp_width(s) <= n:
        return s
    out, w = "", 0
    for c in s:
        cw = 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        if w + cw > n - 1:
            break
        out += c
        w += cw
    return out + "…"


def _verdict(row: dict) -> str:
    """이 행의 결과를 한 눈에 읽히는 판정으로."""
    answerable = row.get("answerable")
    f1 = row.get("f1")
    if answerable is False:  # 문서에 답이 없는 질문 → 기권이 정답
        return "✅ 올바른 기권" if row.get("abstained") else "❌ 환각(답변함)"
    if row.get("abstained"):
        return "⚪ 기권(정보 없음)"
    if f1 is None:
        return "—"
    if row.get("em") == 1.0:
        return f"✅ 정확 (F1 {f1:.2f})"
    if f1 >= 0.6:
        return f"🟢 근접 (F1 {f1:.2f})"
    return f"⚪ 빗나감 (F1 {f1:.2f})"


def _f1_of(row: dict) -> float:
    v = row.get("f1")
    return float(v) if isinstance(v, (int, float)) else 0.0


def _lesson(question_row: dict, by_arm: Dict[str, dict]) -> str:
    if question_row.get("answerable") is False:
        ok = sum(1 for a, _ in _ALL_ARMS if by_arm.get(a, {}).get("abstained"))
        return (
            f"💡 짠! — 문서에 없는 정보는 지어내지 않고 '문서에서 확인할 수 없습니다'라고 "
            f"기권하는 게 정답입니다 ({ok}/{len(_ALL_ARMS)} 모델이 올바르게 기권)."
        )
    best_closed = max((_f1_of(by_arm.get(a, {})) for a, _ in _CLOSED_BOOK), default=0.0)
    best_open = max((_f1_of(by_arm.get(a, {})) for a, _ in _RETRIEVAL), default=0.0)
    if best_open - best_closed >= 0.1:
        return (
            f"💡 짠! — 같은 질문이라도 **검색(retrieval)** 을 켜야 정답이 나옵니다 "
            f"(closed-book 최고 F1 {best_closed:.2f} → 검색 {best_open:.2f}). "
            f"파인튜닝 자체보다 검색이 핵심 — 이게 이 벤치마크의 교훈입니다."
        )
    return f"💡 closed-book 최고 F1 {best_closed:.2f} · 검색 최고 F1 {best_open:.2f}."


# ---------------------------------------------------------------- 질문 해석

def _resolve_qid(catalog: List[dict], question: Optional[str], qa_id: Optional[str]) -> Optional[str]:
    ids = {c["qa_id"] for c in catalog}
    if qa_id:
        return qa_id if qa_id in ids else None
    if not question:
        return _DEFAULT_QID if _DEFAULT_QID in ids else (catalog[0]["qa_id"] if catalog else None)
    if question in ids:  # qa_id를 --question으로 넘긴 경우
        return question
    q = question.strip().lower()
    matches = [c for c in catalog if q in c["question"].lower()]
    return matches[0]["qa_id"] if matches else None


# ---------------------------------------------------------------- 출력

def _print_list(catalog: List[dict]) -> int:
    print(f"📚 사용 가능한 질문 {len(catalog)}개 (public_finance_demo, 합성 한국어 금융 문서)\n")
    for c in catalog:
        tag = "answerable" if c["answerable"] else "unanswerable"
        print(f"  {c['qa_id']:6} [{c['category']:16}·{tag:12}] {c['question']}")
    print("\n예) pdf2llm ask --qa-id q002   |   pdf2llm ask --question \"영업이익\"")
    return 0


def _print_answer(qid: str, all_arms: Dict[str, Dict[str, dict]], seed: int) -> int:
    by_arm = {arm: rows.get(qid, {}) for arm, rows in all_arms.items()}
    meta = next((r for r in by_arm.values() if r), None)
    if meta is None:
        print(f"[ask] '{qid}' 에 대한 기록이 없습니다.", file=sys.stderr)
        return 2

    bar = "═" * 66
    print(bar)
    print("📄 근거 PDF : public_finance_demo (합성 한국어 금융 문서, credential-free)")
    print(f"❓ 질문     : {meta.get('_question', '')}")
    print(f"✅ 정답     : {meta.get('_gold', '') or '문서에서 확인할 수 없습니다.'}")
    print(
        f"   (카테고리 {meta.get('category', '?')} · "
        f"{'답 있음' if meta.get('answerable') else '문서에 없음'} · "
        f"실제 A100 실행, seed {seed})"
    )
    print(bar)

    label_w = 22
    print("\n🔒 Closed-book — 문서 없이 파라미터 기억만으로:")
    for arm, label in _CLOSED_BOOK:
        row = by_arm.get(arm, {})
        if not row:
            continue
        print(f"   {_pad(label, label_w)} {_pad(_verdict(row), 20)} \"{_clip(row.get('_pred'))}\"")

    print("\n🔎 Open-book — 검색(retrieval)으로 관련 문단을 함께 제공:")
    for arm, label in _RETRIEVAL:
        row = by_arm.get(arm, {})
        if not row:
            continue
        crown = " 🏆" if arm == _BEST_ARM else ""
        print(f"   {_pad(label + crown, label_w)} {_pad(_verdict(row), 20)} \"{_clip(row.get('_pred'))}\"")

    print("\n" + _lesson(meta, by_arm))
    print(
        "\n👉 다른 질문:  pdf2llm ask --list   |   "
        "임의 문장 실시간 답변:  pdf2llm ask --live -q \"...\"  (로컬 Ollama)"
    )
    return 0


# ---------------------------------------------------------------- live(선택)

def _ask_live(question: str, model: str) -> int:
    if not question:
        print("[ask --live] -q/--question 으로 질문을 주세요.", file=sys.stderr)
        return 2
    try:
        from pdf_qa.provenance import parse_pdf  # noqa: WPS433
        from .providers import LiveOllamaProvider, ProviderError
    except Exception as exc:  # noqa: BLE001
        print(f"[ask --live] 의존성 로드 실패: {exc}", file=sys.stderr)
        return 1

    texts: List[str] = []
    for name in ("finance_report_v1.pdf", "finance_report_v2.pdf"):
        p = os.path.join(_DOCS, name)
        if os.path.isfile(p):
            doc = parse_pdf(p)
            texts.append("\n".join(getattr(e, "text", "") for e in doc.elements))
    document_text = "\n\n".join(texts)

    print(f"❓ 질문   : {question}")
    print(f"🤖 모델   : Ollama {model} (로컬, 합성 PDF를 근거로)\n")
    try:
        gen = LiveOllamaProvider(model=model).generate(question, document_text)
    except Exception as exc:  # noqa: BLE001 - 데몬 미기동 등
        print(f"[ask --live] Ollama 호출 실패: {exc}", file=sys.stderr)
        print("  → `ollama serve` 실행 후 `ollama pull " + model + "` 를 확인하세요.", file=sys.stderr)
        return 1
    print(f"✅ 답변   : {gen.answer}")
    print("\n(참고: live 경로는 벤치마크 점수화 대상이 아니며, 근거·기권 게이트는 오프라인 파이프라인이 담당합니다.)")
    return 0


# ---------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pdf2llm ask",
        description="학습된 모델에 질문 → 답이 '짠!' (기본: 실제 A100 결과 오프라인 재생)",
    )
    ap.add_argument("-q", "--question", help="질문(부분 문자열 매칭) 또는 qa_id")
    ap.add_argument("--qa-id", help="정확한 qa_id (예: q000)")
    ap.add_argument("--seed", type=int, default=42, help="표시할 seed (기본 42)")
    ap.add_argument("--list", action="store_true", help="사용 가능한 질문 목록 출력")
    ap.add_argument("--live", action="store_true", help="로컬 Ollama로 임의 문장 실시간 답변")
    ap.add_argument("--model", default="qwen2.5:7b-instruct", help="--live 시 Ollama 모델 태그")
    args = ap.parse_args(argv)

    if args.live:
        return _ask_live(args.question or "", args.model)

    catalog = _question_catalog(args.seed)
    if not catalog:
        print(
            "[ask] 벤치마크 결과 파일을 찾지 못했습니다. "
            "먼저 저장소를 클론했는지 확인하세요(historical_final/v1/per_example).",
            file=sys.stderr,
        )
        return 1

    if args.list:
        return _print_list(catalog)

    qid = _resolve_qid(catalog, args.question, args.qa_id)
    if qid is None:
        hint = args.qa_id or args.question
        print(f"[ask] '{hint}' 에 해당하는 질문을 찾지 못했습니다.\n", file=sys.stderr)
        _print_list(catalog)
        return 2

    return _print_answer(qid, _load_all(args.seed), args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
