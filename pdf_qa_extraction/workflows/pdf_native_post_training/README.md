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

## 🧑‍⚖️ Local review workflow (근거 중심 승인 큐)

생성된 Q&A를 **검토 없이 학습에 넣지 않습니다.** 작은 로컬/단일 사용자 리뷰어이므로 의도적으로
**local review workflow**로 부릅니다 — 인증·권한·감사 보존이 없으며 enterprise review system이
아닙니다. 로직: [`review.py`](review.py), 이벤트 스키마: [`schemas/review_event.schema.json`](schemas/review_event.schema.json).

핵심 보증:

- **승인 상태는 append-only 이벤트 로그의 projection** 입니다. JSONL 행의 `review_status` 필드를
  임의로 고쳐도 무시되고, **accept 이벤트가 없으면 unreviewed**로 취급되어 학습에 들어가지 않습니다.
- **기본 학습 export는 `approved`(및 `edited`)만** 포함합니다. reject는 quarantine, 미검토는 held.
  P1-6 source-selection과 **합성**되어 stale/revoked/superseded·충돌 행도 함께 제외됩니다.
- **edit 행은 원본 생성값과 diff를 보존**합니다(`_review_original`/`_review_edits`).
- **모든 approved 행은** reviewer 이벤트 + source evidence로 **역추적**됩니다(`trace`).
- **원문 접근 권한이 없는 report에서는 source snippet을 redact**합니다(`redact_for_report`).

```bash
# 로컬 리뷰 CLI (append-only 이벤트 로그)
python -m workflows.pdf_native_post_training.review --log review.jsonl \
    accept --records gen.jsonl --qa-id q000 --reviewer alice
python -m workflows.pdf_native_post_training.review --log review.jsonl \
    reject --records gen.jsonl --qa-id q003 --reviewer alice --reason ungrounded
python -m workflows.pdf_native_post_training.review --log review.jsonl list --records gen.jsonl
python -m workflows.pdf_native_post_training.review --log review.jsonl \
    export --records gen.jsonl --out review_out/   # train_approved / quarantine / pending
python -m workflows.pdf_native_post_training.review --log review.jsonl verify  # 체인·해시 무결성
```

테스트: `tests/test_review.py`.

## 🔁 Leakage-safe failure-to-data loop (실패 → 데이터)

실패 사례를 데이터로 되돌리는 loop **자체가 harness 증거**입니다. 단, **final test 실패를 학습에
넣고 같은 final test로 개선을 주장하지 않습니다.** 오류 분류: [`evaluation/error_taxonomy.py`](../../evaluation/error_taxonomy.py),
loop 로직: [`failure_to_data.py`](failure_to_data.py).

```text
dev prediction → error taxonomy → source/evidence review
               → approved correction/curriculum row
               → new dataset version → train → dev gate
               → one-time protected current-final evaluation
```

핵심 보증:

- **error taxonomy**는 grounding·wrong-version·numeric/unit·abstention·OCR·citation·schema·policy-violation을
  포함합니다(`classify_error`, reward 컴포넌트 재사용). 리포트는 `summarize`로 집계합니다.
- **failure mining은 dev set만** 사용합니다. non-dev는 `ValueError`, **final ID가 섞이면
  `FinalLeakageError`** 입니다(`mine_failures`).
- **final ID가 correction/export/train에 들어가면 실패**합니다 — `assert_no_final_leakage`가 행의
  `qa_id`와 `derived_from.dev_qa_id`를 모두 검사하고, `build_correction`/`assemble_dataset_version`에
  내장되어 **CI가 깨집니다**(planted-final-ID 테스트가 이를 강제).
- **각 새 training row는** 어떤 dev failure·오류 카테고리·source evidence에서 왔는지 `derived_from`으로
  **역추적**되고, **사람 승인**(review.py 이벤트)을 거칩니다(`build_correction`).
- **동일 dev 반복 최적화 정도**를 `DevReuseLedger`가 라운드별로 기록합니다(overfitting 가시화).
- **final score는 모든 설계 결정 후 한 번** 산출되고 **접근 기록**이 남습니다
  (`FinalAccessLedger.access`/`assert_single_scoring` — 2회 채점 시 예외).

테스트: `tests/test_failure_to_data.py` (카테고리 분류·leakage 차단·lineage·dev-reuse·단일 채점).

## 📊 파이프라인 벤치마크 · 자동 결정 리포트 (raw metrics → decision)

토큰/초만으로는 이 워크플로의 값어치(=**학습 데이터 생산 과정**)를 못 봅니다. 두 도구가 이를 정직하게
측정·의사결정합니다. 벤치마크: [`benchmark_pipeline.py`](benchmark_pipeline.py), 리포트:
[`report.py`](report.py), 통합 스키마: [`schemas/metrics.schema.json`](schemas/metrics.schema.json).

```bash
# 1) 파이프라인을 실제로 돌려 raw metrics 측정 (public demo = CPU, network-free)
python -m workflows.pdf_native_post_training.benchmark_pipeline \
  --config workflows/pdf_native_post_training/configs/demo-replay.yaml \
  --run-dir runs/bench --out runs/bench/pipeline_metrics.json
# 2) 품질·서빙·크기·비용 후보를 제약과 결합해 Pareto frontier + 추천 산출
python -m workflows.pdf_native_post_training.report \
  --decision-config workflows/pdf_native_post_training/configs/decision_constraints.yaml \
  --pipeline-metrics runs/bench/pipeline_metrics.json \
  --out-dir runs/bench/report
```

핵심 보증:

- **동일 스키마**(`pdf2llm-metrics/1`)를 public demo와 GPU run이 공유합니다. 측정 못 한 값은 **`not_measured`**
  (예: CPU의 `peak_vram_mb`), 없는 대상은 `not_applicable`(그림 0개일 때 caption linkage) — **0으로 위장하지
  않습니다.** 각 문서는 파생 원본의 **SHA-256**을 `sources`에 기록합니다.
- **파이프라인 metrics**: pages/sec, element(text/table/figure) throughput, raw→accepted→rejected yield +
  reject 사유, provider 호출·토큰, peak RAM/VRAM, artifact bytes, **evidence pass rate** 등 원자료를 그대로.
  데모 실측: 3 pages / 35 elements / 26→26 accepted(yield 1.0) / evidence_pass 1.0.
- **결정 리포트**: quality·size·VRAM·TTFT/TPOT·throughput·goodput·error-rate·cost를 한 표로 묶고, config의
  제약(예: `peak_vram_gb<=8`, `f1_drop<=1.0`)으로 **feasible 후보 + Pareto frontier**를 계산합니다. 어떤
  후보도 제약을 못 맞추면 **`no_feasible_candidate`** 를 반환합니다(추천을 지어내지 않음).
- **파생값은 전부 코드 산출**이며 사람이 손으로 적지 않습니다. **비용은 출처·시점(`source`/`as_of`)이 있는
  rate card** 로 계산하고, 자기호스팅처럼 per-token 요율이 없으면 `not_measured`.
- **재현 가능한 추천**: 커밋된 예시([`configs/decision_constraints.yaml`](configs/decision_constraints.yaml),
  기록된 A100 v2 실측 3-way 수치)에서 메모리 바운드 서빙 프로파일(≤8 GiB, F1 drop ≤1.0)의
  **decision-report recommendation: `C_int4_qat`** 입니다 — BF16은 VRAM(15.27 GiB)로 탈락, int4 중
  QAT가 PTQ보다 F1이 높아 추천됩니다. `report.py`가 이 문자열을 재계산하며 테스트가 README와 일치를 강제합니다.

테스트: `tests/test_benchmark_pipeline.py`(스키마·데모 카운트·`not_measured` 전파),
`tests/test_report.py`(Pareto·feasibility·`no_feasible_candidate`·rate-card 비용·10-section·README 일치).

## 📐 PDF-native 동일-계약 벤치마크 (public frozen regression) — `results`

KorQuAD quantization 벤치마크(고정 외부 QA)와 **분리된**, 합성 금융 PDF에서 시작해
extraction→dataset→SFT→PTQ/QAT→serving까지 **하나의 metric 계약**으로 비교하는 워크플로-네이티브
벤치마크입니다. 입력·label·evidence를 모두 커밋한 **public frozen regression**이라 `sealed`·`unseen`으로
부르지 않으며, 보호된 current-final 저장소는 운영하지 않아 `final` 슬라이스만 `planned`로 남겨 둡니다.
모델 비교군 6개(Base/SFT/PTQ/QAT ± retrieval)는 **실제 A100에서 정주행**했고(Qwen3-8B, seeds 42/43/44),
raw per-example·자동 summary가 [`historical_final/v1/`](benchmarks/pdf_native/historical_final/v1)에
커밋돼 있습니다. VM은 실행 직후 해제했습니다.

**결과 요약 (mean±std, 3-seed; aggregate는 per_example/*.jsonl에서 자동 재생성):**

| arm | EM | F1 | cite-span | grounded | PII-leak | size GB |
|---|---|---|---|---|---|---|
| base_bf16 (closed-book) | 0.00 | 0.215 | 0.00 | 1.00 | 0.0 | — |
| sft_bf16 (closed-book) | 0.00 | 0.218 | 0.00 | 0.972 | 0.0 | 15.27 |
| sft_int4_ptq (closed-book) | 0.00 | 0.222 | 0.00 | 0.954 | 0.0 | **5.77** |
| sft_int4_qat (closed-book) | 0.00 | 0.215 | 0.00 | 1.00 | 0.0 | **5.77** |
| base_bf16 + retrieval | 0.00 | 0.333 | 0.452 | 0.917 | 0.0 | — |
| **sft_bf16 + retrieval** | **0.226** | **0.444** | 0.452 | 0.861 | 0.0 | 15.27 |

**정직한 해석 (paired bootstrap 95% CI):**
- **Closed-book fine-tuning 효과는 없음**: SFT − Base(둘 다 무맥락) ΔF1 **+0.003**, CI [0.0, 0.024] →
  유의하지 않음. disjoint train family로 SFT해도 eval 사실이 파라미터에 주입되지 않습니다.
- **retrieval이 필요·유효**: Base+retrieval − Base ΔF1 **+0.119**, CI [0.051, 0.198] → **유의**.
- **SFT의 가치는 retrieval과 함께 실현**: SFT+retrieval − Base+retrieval ΔF1 **+0.112**,
  CI [0.056, 0.180] → **유의**. 즉 "실용적 PDF fine-tuning 효과"는 **open-book(검색 포함)에서만** 주장 가능.
- **INT4 양자화는 사실상 무손실 압축**: PTQ/QAT F1 ≈ BF16, 크기 15.27GB→**5.77GB (2.65× 축소)**,
  PII 유출 0, schema 유효율 1.0.
- 작은 N(answerable 31)이라 CI가 넓게 설계돼 있고, mutable 숫자 슬라이스(vf*, N=10)는 모든 arm이 미해결
  (retrieval recall@k≈0.42)입니다. 사전등록 임계값은 결과를 본 뒤에도 **수정하지 않았고**, closed-book
  citation·sft_improves 기준은 설계상 **정직하게 fail**로 기록했습니다(`acceptance.yaml` 참조).

- 위치: [`benchmarks/pdf_native/`](benchmarks/pdf_native/) — `benchmark.yaml`(10개 카테고리 coverage,
  metric 계약, 6개 실험군, 공정비교 조건), `acceptance.yaml`(**실행 전 고정**된 pre-registered 임계값),
  `public_regression.jsonl`(raw set), `final_manifest.json`(set ID·schema·category counts·license·input
  hash·leakage audit·`owner_review_pending`), `splits/`.
- metric 계약 구현: [`evaluation/pdf_native.py`](../../evaluation/pdf_native.py) — EM/F1, numeric/date/unit
  exactness, citation page·span, retrieval recall@k·no-answer rate, abstention P/R, schema validity,
  groundedness, PII leakage, category별 정확도 + failure taxonomy. **aggregate는 raw per-example에서 자동
  생성**됩니다(수기 입력 없음).
- split 원칙: `document_family_id`로 분리하고 v1/v2는 같은 split에 함께 둡니다. `assert_no_split_leakage`가
  family·source-span 0-overlap을 증명해야 publish됩니다.

```bash
# 커밋된 합성 코퍼스에서 벤치마크 산출물을 결정적으로 재생성
python -m workflows.pdf_native_post_training.benchmarks.pdf_native.build_benchmark
```

정직성 가드(spec P1-5): base 모델 arm 없이 SFT/PTQ/QAT 표를 "fine-tuning 효과"라 부르지 않고, retrieval
baseline은 optional이 아니며, 보호 label 없이 `sealed`/`unseen`을 주장하면 claim check가 실패합니다. 테스트:
`tests/test_pdf_native_benchmark.py`(코퍼스 일치·실 fixture leakage 0-overlap·claim-check·pre-registration·
재생성 결정성), `evaluation/tests/test_pdf_native.py`(metric 계약 단위 테스트).

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
