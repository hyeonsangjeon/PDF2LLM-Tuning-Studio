# PDF-native Post-training Workflow

원본 PDF → **근거 주소가 검증된(evidence-grounded)** Q&A 데이터셋 → SFT → 평가로 이어지는 **독립 워크플로**입니다.
공개 저장소를 클론한 사람이 **자격 증명·GPU·네트워크 없이** 전체 파이프라인을 재현할 수 있도록,
합성 한국어 금융 PDF와 **결정적(replay) 생성 캐시**를 함께 제공합니다.

> 이 워크플로는 기존 [`quantization/`](../../quantization/) 트랙(KorQuAD BF16/PTQ/QAT + vLLM 서빙)을 대체하지
> 않습니다. 상류(PDF→dataset→SFT→eval)를 담당하고, 양자화·서빙은 **선택적 하류 단계**로 재사용합니다.

---

## ⏱️ 60초 오프라인 증명 (credential-free proof)

```bash
# 저장소 루트에서 (파이썬 3.10+; 최초 1회 워크플로 의존성 설치)
make install-workflow      # pip install -e "pdf_qa_extraction[workflow]"
make verify-demo           # == pdf2llm verify-demo
```

성공 시 마지막 줄:

```
  [PASS] evidence_address_integrity==1.0
  [PASS] policy_quarantined==0
  [PASS] eval.em==1.0
  [PASS] eval.f1==1.0
  [PASS] train_rows>0
[verify-demo] PASS
```

네트워크·클라우드 키·GPU가 전혀 필요 없고, 실행마다 **동일한 재현 지문(reproducibility fingerprint)** 이 나옵니다.

---

## ✅ Good fit / ⛔ Not a good fit

**Good fit**
- PDF에서 **근거(페이지·요소·인용문)까지 추적되는** 파인튜닝 Q&A를 만들고 싶을 때.
- 클라우드 계정 없이 **오프라인·결정적으로** 전체 파이프라인(수집→생성→검증→정책→큐레이션→학습→평가)을 배우고 싶을 때.
- 로컬 Ollama 등 **자격 증명 불필요** 백엔드로 live 실행을 확장하고 싶을 때.
- 근거 무결성·PII fail-closed 게이트·재현 지문 같은 **신뢰 스캐폴드**를 자신의 데이터에 이식하고 싶을 때.

**Not a good fit (현재 미지원/범위 밖)**
- **프로덕션 서빙/멀티테넌트 API**: 데모 파이프라인이며 SLA·인증·오토스케일 없음.
- **대규모 실학습**: 기본 학습 단계는 CPU 스모크(tiny 모델, 몇 step)로 **메커니즘 증명**용. 실 GPU 학습은 사용자 구성 필요.
- **임의 PDF의 무검증 대량 처리**: 근거 검증은 파서가 해석 가능한 텍스트/표 기준. 스캔 이미지 OCR 정확도는 별도 검증 필요.
- **한국어 외 언어 최적화**: 스코어러 정규화·합성 코퍼스는 한국어 금융 문서 기준.
- **클라우드 egress가 필요한 폐쇄망 규정 준수 주장**: 정책 게이트는 fail-closed 데모이지 인증된 DLP가 아님.

---

## 🚀 Quick start — replay(오프라인)와 live

```bash
# 1) 결정적 오프라인 replay (기본, 키·GPU·네트워크 불필요)
make demo-replay             # pdf2llm demo-replay

# 2) 로컬 Ollama live (실제 모델 생성; Ollama 데몬 필요)
ollama serve &               # 별도 터미널
ollama pull qwen2.5:7b-instruct
make demo-live-ollama        # pdf2llm demo-live-ollama

# 3) CPU SFT 스모크 (아주 작은 모델을 실제로 몇 step 학습)
make demo-train-smoke        # pdf2llm demo-train-smoke

# 특정 config 직접 실행
pdf2llm run --config workflows/pdf_native_post_training/configs/demo-replay.yaml
```

---

## 📤 명령별 기대 출력·리포트 경로

모든 실행은 `runs/<run_id>/` 아래에 아래 구조를 만들고, 콘솔 마지막에 `evidence_address_integrity`,
`policy_quarantined`, `train_rows`, `eval EM/F1`, 단계 상태를 요약합니다.

```
runs/<run_id>/
├── run_manifest.json     # 재현 지문 + 입력/출력 해시 (비밀·절대경로 없음)
├── report.json           # 기계가 읽는 요약 (아래 키)
├── report.md             # 사람이 읽는 요약
├── artifacts/
│   └── train_sft.jsonl   # export된 SFT 학습 행
└── stages/
    ├── ingest.json  generate.json  verify_evidence.json  policy_gate.json
    └── curate.json  export.json    eval.json             report.json
```

`report.json` 최상위 키: `run_id`, `reproducibility_fingerprint`, `mode`, `documents`, `classification`,
`candidates`, `evidence_address_integrity`, `evidence_passed`, `evidence_failed`, `policy_passed`,
`policy_quarantined`, `train_rows_exported`, `eval`.

| 명령 | 성공 시 마지막 줄(요약) | 핵심 생성물 |
|---|---|---|
| `demo-replay` | `stages: {... 'report': 'completed'}` (8단계) | `runs/<id>/report.json`, `artifacts/train_sft.jsonl`(26행) |
| `verify-demo` | `[verify-demo] PASS` | 임시 run-dir(무결성 assert 후 정리) |
| `demo-train-smoke` | 9단계(`train_smoke` 포함) 완료 요약 | `runs/<id>/artifacts/`에 학습 모델 |
| `build-fixture` | 체크섬·gold 개수 출력 | `public_finance_demo/`의 PDF·`gold_qa.jsonl` 재생성 |

기대 기준값(합성 코퍼스): **evidence_address_integrity = 1.0**, **policy_quarantined = 0**,
**train_rows = 26**, replay 모드 **EM/F1 = 1.0**(gold와 recorded 생성이 일치하도록 설계).

---

## 🧰 요구 사항 (측정값 vs 안전 요구사항 구분)

| 항목 | replay / verify-demo | live-ollama | train-smoke |
|---|---|---|---|
| **CPU/GPU** | CPU만 (GPU 불필요) | CPU (모델 실행은 Ollama에 위임) | CPU만 |
| **네트워크** | 불필요(오프라인) | Ollama 로컬 데몬 | 최초 tiny 모델 다운로드 |
| **자격 증명** | 없음 | 없음(로컬) | 없음 |
| **의존성** | `[workflow]`(reportlab·PyMuPDF·jsonschema·pyyaml) | + 실행 중인 Ollama | + `[train]`(torch·transformers·datasets) |
| **디스크** | 수 MB(리포지토리 픽스처 포함) | + 모델 용량(수 GB) | + tiny 모델(수십 MB) |

> 위 표의 CPU/네트워크/자격증명 요구는 실제 실행에서 확인한 값입니다. 디스크/모델 용량은 선택한 모델에 따라
> 달라지므로 **측정하지 않은 최소 사양을 단정하지 않습니다**.

---

## 🩺 Troubleshooting (실패 → 원인 → 해결)

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: reportlab/fitz/jsonschema` | 워크플로 의존성 미설치 | `make install-workflow` (`pip install -e "pdf_qa_extraction[workflow]"`) |
| `egress blocked ... provider` (replay에서) | 정책 게이트가 provider 이름을 로컬로 인식 못함 | `configs/policies/public.yaml`의 `allowed_providers`에 provider 추가(하이픈/언더스코어 무관) |
| live-ollama가 연결 실패 | Ollama 데몬 미기동/모델 미pull | `ollama serve` 후 `ollama pull <model>`; `configs/demo-live-ollama.yaml`의 태그 확인 |
| `evidence_address_integrity < 1.0` | 픽스처 변경 후 gold 인용문이 파서 출력과 불일치 | `pdf2llm build-fixture`로 픽스처·gold 재생성(파서 실출력으로 근거 재해결) |
| train-smoke가 `torch` 없음으로 실패 | `[train]` extra 미설치 | `pip install -e "pdf_qa_extraction[train]"` |
| 재실행이 모두 skip됨 | 내용·config 해시가 동일(정상, resume-by-hash) | 강제 재실행은 새 `--run-dir` 사용 |

---

## 🧭 Stable fact vs mutable fact 분리 (source selection)

금리·한도·수수료·약관·기준일처럼 **바뀌는 값(mutable fact)** 을 model weight에 외우게 하지 않습니다.
mutable fact의 *현재 값*은 retrieval/source selection이 담당하고, SFT는 **인용 형식·근거 없는 답변
거부·계산 절차·답변 구조** 같은 **안정적 behavior(stable)** 학습에 집중합니다.

- 스키마 필드: `fact_volatility(stable|mutable|unknown)`, `document_version`, `supersedes`,
  `source_status(active|stale|revoked|unknown)`, `effective_from/until`,
  `evidence[].document_version` (`pdf_qa/schemas/qa_with_evidence.schema.json`).
- 로직: [`source_selection.py`](source_selection.py) — 순수 파이썬·결정적.
  - `select_source(...)` — 같은 fact의 여러 source 중 **최신 유효본**을 고르거나, 정렬 불가한 충돌·
    revoked-only·**stale mutable** 이면 **abstain**(오래된 값을 자신 있게 답하지 않음).
  - `partition_for_export(...)` — **stale/revoked/superseded 행을 활성 학습 export에서 제외**하고
    versioned archive로 보냅니다(충돌은 review로 held). 파이프라인 export가 이를 강제해
    `report.json`의 `source_partition`(active/versioned_archive/held_for_review)으로 보고합니다.
  - `affected_by_version_change(...)` — 문서 버전 변경 시 영향받는 Q&A와 새 `dataset_version` 추적.
  - `mutable_fact_report(...)` — recency·citation·abstention을 **별도 category**로 집계
    (P1-5의 Base+retrieval vs SFT+retrieval 비교에서 최신성/근거/거부를 분리 보고).
- 공개/합성 fixture: `public_finance_demo/versioned_facts.jsonl` (구·신 버전, 충돌, revoked,
  effective window 케이스). 테스트: `tests/test_source_selection.py`.

## 🧹 Cleanup

- **로컬 실행물**: `runs/`는 `.gitignore`에 포함되어 커밋되지 않습니다. 삭제는 `rm -rf pdf_qa_extraction/runs`.
- **live-ollama**: 받은 모델 정리는 `ollama rm <model>`; 데몬 종료는 해당 프로세스 종료.
- **클라우드/엔드포인트**: 이 워크플로의 데모 경로는 **클라우드 리소스를 만들지 않습니다.** GPU 학습·서빙으로
  확장해 VM/엔드포인트를 만든 경우, 과금 방지를 위해 **사용 직후 반드시 해제**하세요(예: `az group delete`).

---

## 아키텍처 경계 (한 방향 의존)

`workflows/`는 코어(`pdf_qa`)·`evaluation`·`quantization`에 **한 방향으로만** 의존하며, 코어는 절대
`workflows`를 import하지 않습니다. `pdf2llm` 런처는 이 경계를 지키기 위해 워크플로 실행을 **subprocess**로
호출합니다. 이 규칙은 `test/test_architecture_boundary.py`가 강제합니다.
