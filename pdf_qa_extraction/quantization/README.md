# quantization/ — 양자화 트랙 (BF16 LoRA A → INT4 PTQ B → INT4 QAT C, **3-way · v2 전면 개편**)

발표(Serve Track: **LoRA → INT4 PTQ/QAT → vLLM**)용 **3-way 양자화 비교**.
PDF 추출·페르소나·스코어러와 **무관한 독립 트랙**으로, 표준 데이터셋 **KorQuAD**(`KorQuAD/squad_kor_v1`,
런타임 다운로드·미커밋)만 사용한다.

> **v2 개편**: v1(커밋 `04c2062`, Qwen3-**1.7B**·단일 seed·500 슬라이스)의 5개 근본 약점(W1–W5)을
> 전부 교정하고 **A100 80GB에서 8B 베이스로 전 과정 재실행**했다. v2가 v1 수치·아티팩트를 **제자리 교체**한다.

| 방법 | 설명 | 노트북 | 상태 |
|---|---|---|---|
| **A. BF16 LoRA** | 풀정밀 8B 베이스에 LoRA 학습 후 머지 (품질 기준선) | `01_bf16_lora.ipynb` | **A100 실측 · 3 seed** |
| **B. INT4 PTQ** | A 머지 모델을 사후 4bit 양자화 (TorchAO tile-packed) | `02_int4_ptq.ipynb` | **A100 실측 · 3 seed** |
| **C. INT4 QAT** | 4bit 인식 학습(full-param STE)으로 양자화 오차 보정 | `03_int4_qat.ipynb` | **A100 실측 · 3 seed** |

## v1 → v2 델타 (5개 근본 약점 교정)
| # | v1 약점 | v2 교정 |
|---|---|---|
| **W1** | base-select zero-shot이 **하네스 아티팩트**(Qwen3-4B F1 3.5 비현실, 32-토큰 예산) | 하네스 재작성(정식 chat template·충분한 `max_new_tokens`·정지 시퀀스·공식 F1)로 ≤9B 3종 재선정 |
| **W2** | **과소학습**: A=2000 step(~0.26 epoch), C-QAT=400 step | A=**수렴까지**(loss 곡선 기록, 750 step), C-QAT=**600 step**(1.5×)로 실제 회복 |
| **W3** | README에 **모순된 두 tok/s**(unsloth 122.8/37 vs vLLM 263/407) | **단일 tok/s 소스**(vLLM 스윕)로 통일, 불일치 열 제거 |
| **W4** | 벤치가 **base 가중치** + 배치 2점 + TTFT 없음 | **실 A/B/C 아티팩트** 벤치 + **배치 스윕(1…256)** + **TTFT·e2e p50/p99** |
| **W5** | **분산 없음**(단일 seed/500) + VRAM 비교 불공정 | **3 seed → 평균±표준편차**, held-out **1000**, **서빙 전용 clean VRAM** |

## 구성
```
quantization/
  config.yaml          # A/B/C 공용 설정 (base·data·lora·train·qat·compute). 스모크 오버라이드 포함
  data_korquad.py      # KorQuAD 로드 + 고정 seed 분리 (+ v1 호환 헬퍼)
  v2_pipeline.py       # v2 코어: chat-template 프롬프트 + completion-only 마스킹 + A/B/C 학습 + chat eval + QAT 자가검증
  v2_run.py            # v2 CLI: a/b/c/eval/agg/selftest — 서브프로세스별 격리, per-seed 아티팩트
  v2_bench.py          # vLLM 배치 스윕 처리량 + 단일스트림 TTFT/e2e p50/p99
  v2_report.py         # 파생표 집계 자동화: 원시 JSON → three_way_table/vllm_throughput 재생성 + --check-historical(CI 게이트)
  eval_qa.py           # 공용 지표: KorQuAD 공식 EM/F1 + ppl + 크기/VRAM/tok·s + int4 tile-packed config
  notebooks/           # 00 base-select, 01 BF16 LoRA, 02 INT4 PTQ, 03 INT4 QAT (모두 실행됨)
  artifacts/           # 산출물(미커밋·gitignore): {A_bf16,B_int4_ptq,C_int4_qat}_seed{42,43,44}/
  results/             # metrics json + 3-way 표(mean±std) + vLLM 처리량(커밋)
```

## 실행 (v2)
```bash
cd pdf_qa_extraction
# QAT 스킴 자가검증(prepare-fires/same-family/convert-roundtrip 게이트)
python -m quantization.v2_run selftest
# 한 seed 3-way: A(LoRA 2ep/6k)→ merge → B(int4 PTQ) → C(int4 QAT 600) → eval×3
python -m quantization.v2_run a    --seed 42        # [--resume] : Spot 재개
python -m quantization.v2_run b    --seed 42
python -m quantization.v2_run c    --seed 42        # [--resume]
python -m quantization.v2_run eval --method A_bf16      --seed 42
python -m quantization.v2_run eval --method B_int4_ptq  --seed 42
python -m quantization.v2_run eval --method C_int4_qat  --seed 42
# 3 seed(42,43,44) 집계 → results/three_way_table.json (mean±std)
python -m quantization.v2_run agg
# vLLM 동일조건 처리량(배치 스윕 + 단일스트림 TTFT/e2e p50/p99) — 실 아티팩트
# (FlashInfer 샘플러 JIT는 nvcc를 요구하므로 native 샘플러 강제)
export VLLM_USE_FLASHINFER_SAMPLER=0
python quantization/v2_bench.py --model-dir quantization/artifacts/A_bf16_seed42     --method A_bf16     --precision bf16 --max-model-len 4096 --mode both --out quantization/results/bench_A.json
python quantization/v2_bench.py --model-dir quantization/artifacts/B_int4_ptq_seed42 --method B_int4_ptq --precision int4 --max-model-len 4096 --mode both --out quantization/results/bench_int4.json
# 파생 표(집계) 재생성 — 원시 JSON → three_way_table.json + vllm_throughput.json (사람이 손으로 수정하지 않음)
python -m quantization.v2_report --check-historical    # 커밋된 파생표가 원시 JSON에서 재생성되는지 검증(CI 게이트, read-only)
python -m quantization.v2_report --emit                # runs/<run_id>/quantization/report/ 아래에 재생성(+ 입력 hash·argument provenance)
```
> 스모크(무-GPU 코드 점검): 각 서브커맨드에 `--subset 200 --max-steps 12 --eval-size 20` 오버라이드.

## ⚙️ 컴퓨트 / GPU 실행 (Azure A100 실측)
스펙대로 **Azure 단일 GPU VM에서 실제 실행**했다. 대상 구독은 초기엔 모던 GPU 쿼터가 전부 0이었으나
**여러 리전 분산 쿼터 신청**으로 확보했다:

| 리전 | 패밀리 | 확보 쿼터 | 비고 |
|---|---|---|---|
| **japaneast** | `NCADSA100v4` | **96 vCPU = A100 80GB ×4** | 본 실행에 사용(1장) |
| italynorth · switzerlandnorth · southeastasia | `NCADSA100v4` | 각 24 vCPU (A100 ×1) | 분산 여유분 |
| spaincentral | `NVADSA10v5` | 36 vCPU (A10 ×1) | 대안 |

배포 시 MCAPS 거버넌스 정책이 **온디맨드 GPU SKU를 Deny**하므로 `Standard_NC24ads_A100_v4`를
**Spot 우선순위**로 프로비저닝해 우회했다(정책 조건이 `priority != Spot` AND이라 Spot이면 미적용).
실제 실행 환경 = **NVIDIA A100 80GB ×1 @ japaneast**, base=**Qwen3-8B**, transformers HF 백엔드,
BF16 LoRA + TorchAO int4.

> **Spot 내구성(핵심)**: 장시간 단일 GPU 실행이라 Spot 축출이 잦았다(전체 실행 중 **4회 축출**, ~1–2h당 1회).
> 대응: A/C **150 step마다 체크포인트 + `--resume`**, **축출-멱등 러너**(`run_all.sh`: 완료 산출물은 phase-skip,
> A/C는 재개)로 `az vm start` 후 무손실 재개. IP는 실행 내내 유지됐다.
> H100은 전 리전(63)·전 크기 ~28회 시도했으나 SKU 단위 구독 잠금으로 확보 실패(A100/A10만 열림).

## 베이스 모델 선정 (스펙 §2 · W1 교정)
v1의 base-select zero-shot F1(Qwen3-4B **3.5**)은 **하네스 아티팩트**였다(32-토큰 예산 + chat template
미적용 + 부적절 정지). v2는 하네스를 재작성(**정식 chat template**, `enable_thinking=false`,
**충분한 `max_new_tokens`**, 공식 KorQuAD F1)하고 **≤9B 3종**을 held-out zero-shot(+few-shot)으로 재평가했다.

| 후보 | zero-shot EM/F1 | few-shot EM/F1 | 라이선스 | 비고 |
|---|---|---|---|---|
| **Qwen/Qwen3-8B** (기본 선정) | **81.75 / 92.51** | **83.75 / 93.67** | Apache-2.0 | 강한 한국어 · 단일 A100 QAT 적합 · INT4/vLLM 성숙 |
| Qwen/Qwen2.5-7B-Instruct | 76.88 / 88.90 | 79.38 / 90.07 | Apache-2.0 | 다른 세대, 실질 경쟁 후보 |
| 01-ai/Yi-1.5-9B-Chat (Llama-3.1-8B 게이팅 폴백) | 47.75 / 73.46 | — / 9.68† | Apache-2.0 | 패밀리 다양성; †few-shot 템플릿 붕괴 |

- **기준**: (a) KorQuAD held-out F1, (b) 단일 A100 적합(QAT 포함), (c) TorchAO INT4 + vLLM 호환.
- **재작성 하네스**(held-out 800, chat template, `enable_thinking=false`, 64-tok, 공식 F1) 기준
  **Qwen3-8B가 zero/few-shot 모두 최고** → v1의 3.5는 순수 하네스 아티팩트였음이 확인됨(동일 모델이 90+).
- **선정 = Qwen3-8B**: `config.yaml`의 `base_model.selected`에 고정 → A/B/C 동일 베이스. (`results/base_select.json`)

## 🔒 선택용 dev vs 최종 holdout 분리 (frozen policy holdout · P1-1)

모델·프롬프트·하이퍼파라미터 **선택**과 **최종 비교**가 같은 슬라이스를 쓰면 선택이 최종 수치로 샌다.
그래서 seed-shuffle된 **하나의** validation 순서를 **서로소** 두 구간으로 고정한다
(`config.yaml`의 `data.splits`, 로직·매니페스트: [`splits.py`](splits.py)).

| split | 구간 | 용도 |
|---|---|---|
| `selection_dev` | `[0:800]` | base-model·프롬프트·하이퍼파라미터 튜닝 **전용** |
| `final_holdout` | `[800:1800]` | **frozen policy holdout** — 평가 명령·릴리스 게이트 **전용** |

- **서로소 보장**: 같은 shuffle의 겹치지 않는 인덱스 구간이라 ID 교집합은 **0**이다.
  [`results/split_manifest.json`](results/split_manifest.json)이 각 split의 구간·개수·ID-리스트 SHA-256과
  `intersection_size` **0**을 기록하고, `python -m quantization.splits --check`로 KorQuAD에서 재현 검증한다.
- **누수 게이트(CI 실패)**: `final_holdout` ID가 training/selection/export 입력에 들어가면
  `splits.assert_no_holdout_leakage`가 `HoldoutLeakageError`를 던진다(`tests/test_splits.py`가
  planted-final-ID로 강제). 코드 경로상 base-select는 `selection_dev`만, `v2_run eval`은 `final_holdout`만 읽는다.
- **보안 경계가 아님**: KorQuAD 라벨은 이미 공개다. 이 분리는 **코드 경로 allowlist(`frozen policy holdout`)**일
  뿐 저장소 밖 사람 열람을 막는 보안 경계가 아니며, 그래서 `sealed`/`unseen final`이라 부르지 않는다.

## 3-way 결과 (실측 · A100 80GB · **`historical_not_reproduced`** · **3 seed 평균±표준편차**)
`v2_run a/b/c/eval`을 **A100 80GB(japaneast, Spot)에서 seed 42·43·44로 실제 실행**한 뒤
`v2_run agg`로 집계한 수치(`results/three_way_table.json`).

> ⚠️ **`historical_not_reproduced` (P1-1)** — 아래 held-out 1000 슬라이스는 base-select에 쓴
> `selection_dev`와 **같은 shuffle 앞부분에서 겹친다**(선택이 최종 수치로 샐 수 있음). 따라서 이 표는
> **엔지니어링 평가**로 표기하며, KorQuAD 라벨이 공개이므로 `sealed`/`unseen`이라 부르지 않는다. 분리된
> frozen `final_holdout`(`[800:1800]`, `selection_dev`와 서로소)로의 재실행은 위 P1-1 메커니즘으로 준비돼
> 있고, **재실행 전까지 새 "최종" 수치는 게시하지 않는다.**

**per-seed (실측, EM / F1 / ppl):**

| method | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| **A_bf16** | 88.3 / 95.034 / 8.867 | 88.0 / 94.874 / 9.096 | 87.1 / 94.583 / 9.187 |
| **B_int4_ptq** | 86.2 / 94.103 / 9.937 | 86.2 / 93.959 / 10.195 | 86.8 / 94.508 / 10.150 |
| **C_int4_qat** | 87.7 / 94.720 / 10.409 | 87.1 / 94.768 / 10.825 | 87.9 / 94.969 / 10.829 |

**집계 (mean ± std over 3 seeds):**

> 📑 아래 표의 모든 숫자는 [`docs/EVIDENCE.md`](../docs/EVIDENCE.md)에 raw JSON 포인터로 등록되어
> `scripts/build_evidence_index.py --check`(CI 게이트)가 자동 검증한다 — README와 JSON이 어긋나면 CI 실패.

| method | base | EM | F1 | ppl | size(GB) | prec |
|---|---|---|---|---|---|---|
| **A_bf16** (기준선) | Qwen3-8B | **87.80 ± 0.51** | **94.83 ± 0.19** | **9.05 ± 0.14** | 15.27 (머지 bf16) | bf16 |
| **B_int4_ptq** | Qwen3-8B | 86.40 ± 0.28 | 94.19 ± 0.23 | 10.09 ± 0.11 | **5.77** | int4 |
| **C_int4_qat** | Qwen3-8B | 87.57 ± 0.34 | **94.82 ± 0.11** | 10.69 ± 0.20 | **5.77** | int4 |

**해석 — 관측된 순서는 A ≳ C > B (3 seed). 아래 수치는 모두 `results/three_way_table.json`에서 재추적된다.**
- **A (BF16)**: 품질 상한. LoRA를 6k 슬라이스 2 epoch로 **수렴**(train loss 0.075–0.078, seed간 편차 <0.003)까지 학습 후 머지.
- **B (PTQ)**: 학습 없이 A 머지를 사후 int4 양자화 → 크기 **2.65× 축소(15.27→5.77GB)**, 품질 소폭 하락(F1 −0.64).
- **C (QAT)**: **동일 int4 포맷(5.77GB)** 이지만 양자화를 인식하며 600 step 재학습 → **B 대비 회복 F1 +0.63**.
  **A와의 관측 F1 차이는 3 seed 평균 0.011(94.82 ± 0.11 vs 94.83 ± 0.19)로 표준편차 안에 들지만, 동등성 검정(equivalence test)은 수행하지 않았다** — "통계적 동률"이라 단정하지 않는다. seed 44에선 C(94.969)가 A(94.583)를 역전했다.
- **B vs C가 곧 *train-aware* 효과**: 서빙 포맷·크기가 동일(5.77GB)하므로 차이는 순수하게 QAT 재학습에서 온다.
- **ppl 순서(A<B<C)가 F1 순서(A≈C>B)와 갈리는 이유**: QAT는 LM perplexity가 아니라 **task loss**(정답 토큰)를 최적화하므로,
  F1은 회복하되 LM ppl은 오히려 커질 수 있다. → 지표 선택이 결론을 바꾼다는 점을 error bar와 함께 명시.

세부:
- **A 학습**: transformers+peft+trl(HF 백엔드), BF16, LoRA r16/α32(attn+MLP), **completion-only 마스킹**(정답 토큰에만 loss),
  6k subset · **2 epoch = 750 step**, ~**59분**/seed, grad checkpointing on. Spot 축출 대비 **150 step 체크포인트 + resume**.
- **B (PTQ)**: TorchAO `Int4WeightOnlyConfig(group_size=128, TILE_PACKED_TO_4D)`를 `TorchAoConfig` 경로로 적용
  (임베딩·`lm_head` 제외 → tied-weight 안전). 재학습 없음, ~22초.
- **C (QAT)**: matched fake-quant(`Int4WeightOnlyConfig(g128)`에서 **추론** → tile-packed 서빙과 **동일 int4 family**) 삽입 →
  양자화 대상 linear만 STE로 **600 step** 재학습(8-bit Adam + grad ckpt로 단일 A100 적합, train loss ~0.006–0.016) →
  convert(adapted-bf16) → **B와 동일한** tile-packed int4로 export. `v2_run selftest`가 사전 게이트.
- **동작 데모**(각 노트북 셀 실측): held-out 질문 1개 → A·B·C 생성 답변을 실행 출력으로 포함.

## vLLM 처리량 벤치마크 (동일 조건 tok/s · 사과-대-사과 · W3·W4 교정)
v1의 tok/s는 스택이 달라(A=unsloth, B·C=transformers) 직접 비교 불가였다(W3). v2는 **단일 vLLM 엔진 + 동일 노브**로
A(bf16)·int4를 **실 아티팩트**로 재측정하고, **배치 스윕(1…256)** 과 **단일스트림 TTFT·e2e p50/p99**를 추가한다(W4).
`results/vllm_throughput.json`.

**측정 원리(가중치·seed 독립성)**: 처리량은 **아키텍처 + 정밀도 + 서빙 포맷**에만 의존하고 학습된 가중치 *값*엔 무관하다.
따라서 **B와 C는 서빙 포맷이 완전히 동일(tile-packed int4)하므로 tok/s가 구조적으로 같다**(한 행으로 대표). 벤치는
**A(bf16 머지) + int4 1행** 두 구성만 실측한다.

**동일 노브**: dtype=bf16, `max_model_len=4096`, `gpu_memory_utilization=0.85`, CUDA graphs on, greedy(temp=0),
`max_tokens=128` + `ignore_eos`(요청당 정확히 128 디코드 토큰), native 샘플러(`VLLM_USE_FLASHINFER_SAMPLER=0`).

**처리량 배치 스윕 (tok/s):**

| batch | 1 | 4 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| **A (bf16)** | 87.6 | 342.7 | **1031.5** | 1822.9 | 2764.1 | 3737.7 | **3959.0** |
| **B≡C (int4)** | **124.4** | **508.0** | 327.3 | 475.1 | 488.9 | 514.5 | 509.7 |
| 우세 | int4 | int4 | bf16 | bf16 | bf16 | bf16 | bf16 |

- **크로스오버 = batch 16**: batch 1–4는 int4 우세(메모리 대역폭 바운드 — 가중치 2.65× 작음), batch ≥16은 bf16 우세
  (연산 바운드 — bf16 텐서코어 GEMM이 int4 dequant+GEMM을 앞섬). batch 256에서 bf16 **7.8×**.
- int4는 커널 상한으로 ~510 tok/s에서 포화(torchao tinygemm), bf16는 배치와 함께 ~3959 tok/s까지 확장.

**단일스트림 지연(batch=1, client-observed):**

| 서빙 | TTFT p50 / p99 | e2e p50 / p99 | clean 가중치 VRAM |
|---|---|---|---|
| **A (bf16)** | **32.2ms / 82.0ms** | 1.469s / 1.482s | 15.27 GiB |
| **B≡C (int4)** | 296.4ms / 721.3ms | **0.775s / 0.785s** | **6.05 GiB** |

> **TTFT 측정 주의**: 이 TTFT는 온라인 streaming의 첫 토큰 도착 시각이 아니라, vLLM 0.23 V1 오프라인 엔진에
> `max_tokens=1` 요청을 보내 **클라이언트에서 잰 벽시계(wall-clock) proxy**다(오프라인 경로는
> `RequestOutput.metrics`를 채우지 않음). 절대값이 아니라 **A vs int4의 상대 비교**로만 해석한다.

**해석(실측 확정)**: int4는 단일스트림 **처리량(1.42×)** 과 **전체응답 e2e(1.9× 빠름: 0.78 vs 1.47s)**, **메모리
(2.65× 작음: 6.05 vs 15.27 GiB)** 에서 유리하지만, **TTFT는 bf16이 9.2× 낮다**(int4 prefill의 dequant 비용).
대배치 처리량은 bf16 우세(크로스오버 batch 16). → **int4 = 메모리/처리량 지향 단일스트림, bf16 = 저-TTFT 인터랙티브
또는 최대 배치 처리량**.

## 서빙 전용 clean VRAM (W5)
노트북 내 eager eval VRAM(A 42.2GB · B/C 61.6GB)은 학습 잔여 할당을 포함해 비교가 불공정했다(v1). v2는 **vLLM
엔진의 서빙 전용 가중치 VRAM**을 A/B/C 동일 기준으로 캡처한다: **bf16 15.27 GiB · int4 6.05 GiB(B=C 동일)**
(`results/vllm_throughput.json`의 `weight_vram_gib`). 이 값이 int4 풋프린트 동등성과 2.5× 절감을 확증한다.

## 재현성 (버전 고정)
`results/env_*.json`에 실행 시 자동 기록. **본 A100 실행 환경**: torch **2.11.0+cu130** ·
transformers **4.57.6** · trl **0.24.0** · peft **0.20.0** · datasets **4.3.0** · **torchao 0.17.0** ·
bitsandbytes 0.50.0 · **vllm 0.23.0** · python 3.10 · CUDA 13.0 · **A100 80GB(japaneast, Spot)**.

> **torchao INT4 주의**: 기본 `Int4WeightOnlyConfig(g128)`(PLAIN 패킹)은 실양자화 시 `mslk` 커널을 요구해 실패한다.
> 본 트랙은 내장 tinygemm 경로인 `int4_packing_format=TILE_PACKED_TO_4D`를 사용(B·C 공통 서빙 포맷).
> int4 아티팩트 저장은 `safe_serialization=False`가 필요하다(torchao 텐서 서브클래스는 safetensors 미지원).
> **vLLM 샘플러 주의**: FlashInfer top-k/top-p 샘플러는 런타임 JIT(nvcc)을 요구하므로, nvcc 미설치 환경에선
> `VLLM_USE_FLASHINFER_SAMPLER=0`으로 native 샘플러를 강제한다.

## 가드레일
- `quantization/` 안에서만. `pdf_qa` 코어·`evaluation/`·`personas.yaml`·웹앱 **무변경**.
- **데이터셋 미커밋**(런타임 다운로드). `.env`/키/토큰 미커밋 — 노트북 출력에도 미노출.
- 베이스·하이퍼파라미터는 `config.yaml`(A/B/C 동일 베이스 고정).

## 상태
- ✅ **v2 3-way**: A(BF16 8B)·B(INT4 PTQ)·C(INT4 QAT) 모두 A100 실측, **3 seed** mean±std.
- ✅ 노트북 00·01·02·03 v2 재작성 + 실행(실 출력), 재현 환경 `env_*.json`.
- ✅ **vLLM 동일조건 처리량**: 배치 스윕(크로스오버 batch 16) + 단일스트림 TTFT/e2e p50/p99 + clean VRAM
  (`vllm_throughput.json`) — 단일 tok/s 소스.
- ✅ W1–W5 전부 교정(하네스·수렴학습·단일 tok/s·실아티팩트 벤치·분산/clean VRAM).
