"""``pdf2llm publish-hf`` — 파인튜닝된 가중치를 HuggingFace Hub에 올린다.

``make bench`` (또는 ``run_arms.py --keep-artifacts``) 로 만들어진 arm 디렉터리(예:
``artifacts/sft_bf16_seed42``, 자립형 merged 16-bit 모델)를 받아, 벤치마크 ``summary.json``
에서 뽑은 점수로 **모델 카드**를 자동 생성하고 저장소를 만들어 업로드한다.

업로드가 끝나면 누구나 아래 한 줄로 **그 가중치를 실제 로드해 실시간 추론**할 수 있다::

    pdf2llm ask --hf <your-name>/<repo> -q "2024년 연간 매출액은 얼마입니까?"

토큰 없이 카드/파일 목록만 확인하려면 ``--dry-run`` 을 쓴다(네트워크·인증 불필요).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SUMMARY = os.path.join(
    _HERE, "benchmarks", "pdf_native", "historical_final", "v1", "summary.json"
)
# 카드에 함께 보여줄 대표 arm (closed-book vs retrieval 대비를 위해 둘 다 인용).
_CLOSED_ARM = "sft_bf16"
_RETRIEVAL_ARM = "sft_bf16_retrieval"
_MODEL_FILE_HINTS = ("config.json",)


def _load_summary(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _metric(summary: Dict[str, Any], arm: str, key: str) -> Optional[float]:
    m = (((summary.get("arms") or {}).get(arm) or {}).get("metrics") or {}).get(key) or {}
    v = m.get("mean")
    return float(v) if isinstance(v, (int, float)) else None


def _fmt(v: Optional[float], nd: int = 3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "n/a"


def build_model_card(summary: Dict[str, Any], *, repo_id: str, base_model: str,
                     arm: str) -> str:
    """summary.json 의 실측 점수로 정직한 모델 카드(Markdown + YAML front-matter)를 만든다."""
    seeds = summary.get("seeds") or []
    f1_closed = _metric(summary, _CLOSED_ARM, "f1")
    f1_retr = _metric(summary, _RETRIEVAL_ARM, "f1")
    em_retr = _metric(summary, _RETRIEVAL_ARM, "em")
    ground = _metric(summary, _RETRIEVAL_ARM, "groundedness_rate")
    pii = _metric(summary, _RETRIEVAL_ARM, "pii_leakage_rate")
    size = _metric(summary, arm, "size_gb")

    front = [
        "---",
        "license: apache-2.0",
        f"base_model: {base_model}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- pdf2llm",
        "- korean",
        "- finance-qa",
        "- lora-sft",
        "- retrieval",
        "---",
    ]
    body = f"""
# {repo_id}

**PDF2LLM-Tuning-Studio** 6-arm 벤치마크의 `{arm}` 파인튜닝 가중치입니다.
`{base_model}` 를 합성 한국어 금융 QA 코퍼스로 LoRA SFT 후 16-bit 병합(merged)한 자립형 모델로,
`AutoModelForCausalLM.from_pretrained("{repo_id}")` 로 바로 로드됩니다.

## 바로 써보기 (실제 가중치 로드 → 실시간 추론)

```bash
pip install "pdf2llm[workflow,train] @ git+https://github.com/hyeonsangjeon/PDF2LLM-Tuning-Studio"
pdf2llm ask --hf {repo_id} -q "2024년 연간 매출액은 얼마입니까?"
```

`ask --hf` 는 재생(replay)이 아니라 이 가중치를 실제로 로드해 `transformers` 로 생성하며,
벤치마크와 **동일한** chat 프롬프트·BM25 검색·정답 추출을 재사용합니다.

## 측정된 성능 (실제 A100, seeds={seeds})

| 조건 | F1 | EM | groundedness | PII-leak |
|---|---|---|---|---|
| closed-book (`{_CLOSED_ARM}`) | {_fmt(f1_closed)} | — | — | — |
| **+ 검색 (`{_RETRIEVAL_ARM}`)** | **{_fmt(f1_retr)}** | {_fmt(em_retr)} | {_fmt(ground)} | {_fmt(pii)} |

> **핵심 교훈:** closed-book 파인튜닝만으로는 사실이 주입되지 않습니다(F1 {_fmt(f1_closed)}).
> **검색(retrieval)을 켜야** 점수가 오릅니다(F1 {_fmt(f1_retr)}). 이 모델은 반드시 문맥과 함께 쓰세요
> — `pdf2llm ask --hf` 가 그 검색을 자동으로 붙여줍니다.

## 정직한 한계

- 학습·평가 데이터는 **합성(synthetic)** 금융 문서입니다(실제 기업 정보 아님).
- 파라미터 크기 ≈ {_fmt(size, 2)} GB. 8B 로딩은 GPU(또는 넉넉한 RAM)를 권장합니다.
- 처음부터 재현: 저장소의 `make bench` (자기 GPU에서 학습→평가→가중치 생성).

_모델 카드는 저장소 `summary.json` 실측값에서 자동 생성되었습니다._
"""
    return "\n".join(front) + "\n" + body.lstrip("\n")


def _looks_like_model_dir(model_dir: str) -> bool:
    return any(os.path.isfile(os.path.join(model_dir, h)) for h in _MODEL_FILE_HINTS)


def _list_files(model_dir: str) -> List[str]:
    out: List[str] = []
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            out.append(os.path.relpath(os.path.join(root, f), model_dir))
    return sorted(out)


def publish(model_dir: str, repo_id: str, *, base_model: str, arm: str,
            summary_path: str, token: Optional[str] = None, private: bool = False,
            dry_run: bool = False, write_card: bool = True) -> int:
    if not os.path.isdir(model_dir):
        print(f"[publish-hf] 모델 디렉터리가 없습니다: {model_dir}", file=sys.stderr)
        return 1

    summary = _load_summary(summary_path)
    base_model = base_model or summary.get("base_model") or "Qwen/Qwen3-8B"
    card = build_model_card(summary, repo_id=repo_id, base_model=base_model, arm=arm)

    if dry_run:
        print("=== [dry-run] 생성될 모델 카드 (업로드 안 함) ===\n")
        print(card)
        print("\n=== [dry-run] 업로드 대상 파일 ===")
        files = _list_files(model_dir)
        if not files:
            print("  (비어 있음 — 실제로는 학습 산출물 디렉터리를 지정하세요)")
        for f in files:
            print(f"  {f}")
        if not _looks_like_model_dir(model_dir):
            print("\n⚠  config.json 이 없어 보입니다. merged 모델 디렉터리인지 확인하세요.",
                  file=sys.stderr)
        print(f"\n[dry-run] 실제 업로드하려면 --dry-run 을 빼고 HF_TOKEN 을 설정하세요 → {repo_id}")
        return 0

    if not _looks_like_model_dir(model_dir):
        print(f"[publish-hf] {model_dir} 에 config.json 이 없습니다 — merged 모델 디렉터리를 "
              "지정하세요(예: artifacts/sft_bf16_seed42).", file=sys.stderr)
        return 1

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[publish-hf] HF 토큰이 없습니다. `export HF_TOKEN=...` 또는 --token 을 주세요 "
              "(write 권한 필요).", file=sys.stderr)
        return 2
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # noqa: BLE001
        print(f"[publish-hf] huggingface_hub 가 필요합니다: {exc}", file=sys.stderr)
        return 1

    if write_card:
        with open(os.path.join(model_dir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(card)

    api = HfApi(token=token)
    print(f"[publish-hf] 저장소 생성/확인: {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"[publish-hf] 업로드 중… {model_dir} → {repo_id}")
    api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type="model")
    print(f"✅ 업로드 완료 → https://huggingface.co/{repo_id}")
    print(f'   이제 어디서나:  pdf2llm ask --hf {repo_id} -q "2024년 연간 매출액은 얼마입니까?"')
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pdf2llm publish-hf",
        description="파인튜닝된 arm 가중치를 HuggingFace Hub에 업로드 (모델 카드 자동 생성)",
    )
    ap.add_argument("--model-dir", required=True,
                    help="업로드할 학습 산출물 디렉터리 (예: artifacts/sft_bf16_seed42)")
    ap.add_argument("--repo-id", required=True, help="대상 HF 저장소 (예: your-name/pdf2llm-sft-qwen3-8b)")
    ap.add_argument("--arm", default=_RETRIEVAL_ARM, help=f"카드 점수 기준 arm (기본 {_RETRIEVAL_ARM})")
    ap.add_argument("--base-model", default=None, help="베이스 모델 (기본: summary.json 값)")
    ap.add_argument("--summary", default=_DEFAULT_SUMMARY, help="점수 출처 summary.json 경로")
    ap.add_argument("--token", default=None, help="HF write 토큰 (기본: $HF_TOKEN)")
    ap.add_argument("--private", action="store_true", help="비공개 저장소로 생성")
    ap.add_argument("--dry-run", action="store_true", help="업로드 없이 카드/파일 목록만 출력(토큰 불필요)")
    ap.add_argument("--no-card", action="store_true", help="README.md 모델 카드 작성/포함 생략")
    args = ap.parse_args(argv)

    return publish(args.model_dir, args.repo_id, base_model=args.base_model, arm=args.arm,
                   summary_path=args.summary, token=args.token, private=args.private,
                   dry_run=args.dry_run, write_card=not args.no_card)


if __name__ == "__main__":
    raise SystemExit(main())
