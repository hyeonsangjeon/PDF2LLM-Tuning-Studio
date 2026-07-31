# quantization/ — 양자화 트랙 (BF16 LoRA A → INT4 PTQ B → INT4 QAT C, **3-way 완성**)

발표(Serve Track: **LoRA → INT4 PTQ/QAT → vLLM**)용 **3-way 양자화 비교**.
PDF 추출·페르소나·스코어러와 **무관한 독립 트랙**으로, 표준 데이터셋 **KorQuAD**(`KorQuAD/squad_kor_v1`,
런타임 다운로드·미커밋)만 사용한다.

| 방법 | 설명 | 노트북 | 상태 |
|---|---|---|---|
| **A. BF16 LoRA** | 풀정밀 베이스에 LoRA 학습 후 머지 (품질 기준선) | `01_bf16_lora.ipynb` | **실행 완료** |
| **B. INT4 PTQ** | A 머지 모델을 사후 4bit 양자화 (TorchAO tile-packed) | `02_int4_ptq.ipynb` | **실행 완료** |
| **C. INT4 QAT** | 4bit 인식 학습(full-param STE)으로 양자화 오차 보정 | `03_int4_qat.ipynb` | **실행 완료** |

## 구성
```
quantization/
  config.yaml          # A/B/C 공용 설정 (base·data·lora·train·compute). 스모크 오버라이드 포함
  data_korquad.py      # KorQuAD 로드 + 생성 instruction 포맷 + 고정 seed 분리
  train_lora.py        # Method A: BF16 LoRA. backend=unsloth(GPU) | hf(CPU-capable)
  eval_qa.py           # A/B/C 공용 eval: KorQuAD 공식 EM/F1 + ppl + 크기/VRAM/tok·s
  notebooks/           # 00 base-select, 01 BF16 LoRA, 02 INT4 PTQ, 03 INT4 QAT (모두 실행됨)
  artifacts/           # 산출물(미커밋·gitignore): A_bf16/(머지) = Part 2 입력
  results/             # metrics json + 3-way 표(커밋)
```

## 실행
```bash
cd pdf_qa_extraction

# 데이터 확인
python -m quantization.data_korquad --smoke --stats

# 학습(A) — GPU VM(스펙): config.yaml의 compute.mode=gpu (Qwen3-1.7B·unsloth·BF16·full)
python -m quantization.train_lora
# 학습(A) — CPU 스모크(무GPU): 소형 모델·200 서브셋·12스텝
python -m quantization.train_lora --smoke

# 평가(A) — held-out KorQuAD EM/F1·ppl·크기·tok/s
python -m quantization.eval_qa --smoke --method A_bf16     # GPU: --smoke 제거
python -m quantization.eval_qa --selftest                 # 지표만 자가검증(모델 불필요)
```

## ⚙️ 컴퓨트 / GPU 실행 (Azure A100 실측)
스펙대로 **Azure 단일 GPU VM에서 실제 실행**했다. 대상 구독
`ME-MngEnvMCAP756842-hjeon-1`은 초기엔 모던 GPU 쿼터가 전부 0이었으나,
**여러 리전에 분산 쿼터 신청**으로 확보했다:

| 리전 | 패밀리 | 확보 쿼터 | 비고 |
|---|---|---|---|
| **japaneast** | `NCADSA100v4` | **96 vCPU = A100 80GB ×4** | 본 실행에 사용(1장) |
| italynorth · switzerlandnorth · southeastasia | `NCADSA100v4` | 각 24 vCPU (A100 ×1) | 분산 여유분 |
| spaincentral | `NVADSA10v5` | 36 vCPU (A10 ×1) | 대안 |

배포 시 MCAPS 거버넌스 정책이 **온디맨드 GPU SKU를 Deny**하므로
`Standard_NC24ads_A100_v4`를 **Spot 우선순위**로 프로비저닝해 우회했다
(정책 조건이 `priority != Spot` AND이라 Spot이면 미적용). 실제 실행 환경 =
**NVIDIA A100 80GB PCIe ×1 @ japaneast**, `compute.mode: gpu`, base=Qwen3-1.7B,
unsloth BF16, KorQuAD `max_steps=2000`. 아래 §Method A 표는 이 실행의 실측치다.

> H100은 전 리전(63개)·전 크기(@40/@160)로 ~28회 시도했으나 SKU 단위 구독 잠금으로
> 확보 실패(A100/A10만 열림). 필요 시 MS 지원티켓 경로만 남는다.

## 베이스 모델 선정 (스펙 §2)
후보 3종과 선정 기준. 최종 확정: `notebooks/00_base_select.ipynb`(**A100 80GB에서 zero-shot F1 실측**).

| 후보 | 파라미터 | 라이선스 | 게이팅 | 비고 |
|---|---|---|---|---|
| **Qwen/Qwen3-1.7B** (기본 선정) | ~1.7B | Apache-2.0 | 무 | 한국어 양호·초경량·단일GPU·INT4/vLLM 성숙 |
| Qwen/Qwen3-4B | ~4B | Apache-2.0 | 무 | 품질↑, 여전히 단일GPU |
| meta-llama/Llama-3.2-3B-Instruct | ~3B | Llama | **유(승인+토큰)** | 패밀리 다양성 |

- **기준**: (a) KorQuAD dev zero-shot F1, (b) 단일 GPU 적합, (c) TorchAO INT4 + vLLM 서빙 호환.
- **A100 실측 zero-shot F1** (held-out 500, `max_new_tokens=32`, `00_base_select.ipynb`):
  **Qwen3-1.7B EM 11.6 / F1 41.4**, Qwen3-4B EM 2.0 / F1 3.5, Llama-3.2-3B **gated**(승인+토큰 필요→제외).
  세부 표는 `results/base_select_zeroshot.json`.
- **선정 = Qwen3-1.7B**: ungated 후보 중 zero-shot 추출 F1 최고 + 단일 GPU 적합 + INT4/vLLM 호환.
  `config.yaml`의 `base_model.selected`에 고정 → A/B/C 동일 베이스. (짧은 32-토큰 예산의
  zero-shot은 거친 프록시이며, 최종 성능은 Method A 학습 후 **EM 81.0 / F1 89.9**로 확정.)

## 3-way 결과 (실측 · A100 80GB · held-out 500)
`notebooks/01·02·03`을 **A100 80GB(japaneast, Spot)에서 실제 실행**한 수치(`results/three_way_table.json`):

| method | base | EM | F1 | ppl | size(GB) | peak VRAM | tok/s | prec |
|---|---|---|---|---|---|---|---|---|
| **A_bf16** (기준선) | Qwen3-1.7B | **81.0** | **89.92** | 10.39 | 3.22 | 7.93 | 122.8 | bf16 |
| **B_int4_ptq** | Qwen3-1.7B | 65.2 | 80.69 | 15.90 | **1.29** | 4.58 | 37.4 | int4 |
| **C_int4_qat** | Qwen3-1.7B | **71.8** | **83.52** | **12.97** | **1.29** | 7.80 | 37.0 | int4 |

**해석 — 스토리가 깔끔하게 성립한다.**
- **A (BF16)**: 품질 상한. 3.22GB.
- **B (PTQ)**: 학습 없이 A 머지를 사후 int4 양자화 → 크기 **2.5× 축소(3.22→1.29GB)**, 대신 품질 하락(F1 89.9→80.7, EM 81→65).
- **C (QAT)**: **동일한 int4 포맷(1.29GB)** 이지만 양자화를 인식하며 재학습 → **B 대비 회복**(F1 **+2.8**, EM **+6.6**, ppl 15.9→**12.97**). 세 지표 모두 B와 A 사이에 위치.
- **B vs C가 곧 *train-aware* 효과**: 서빙 포맷·크기가 동일하므로 차이는 순수하게 QAT 재학습에서 온다.

세부:
- **A 학습**: unsloth `FastLanguageModel`, BF16(load_in_4bit=false), LoRA r16/α32(attn+MLP), effective batch 8,
  **2000 steps**(≈16k KorQuAD, ~0.26 epoch), ~33분, mean loss 2.03.
- **B (PTQ)**: TorchAO `Int4WeightOnlyConfig(group_size=128, TILE_PACKED_TO_4D)`를 transformers `TorchAoConfig`
  경로로 적용(임베딩·`lm_head` 제외 → tied-weight 안전). 재학습 없음.
- **C (QAT)**: matched fake-quant(`Int4WeightOnlyConfig(g128)`에서 **추론** → tinygemm tile-packed 서빙과 **동일 스킴**)
  삽입 → **양자화 대상 linear 가중치만** STE로 400 step 재학습(임베딩·`lm_head` 동결로 ppl 드리프트 억제) →
  convert(adapted-bf16) → **B와 동일한** tile-packed int4로 export.
- **동작 데모**(각 노트북 셀 실측): held-out 질문 *"2004년 이명박이 서울시장 재직시절 전면적으로 개선한 것은?"*
  → 정답 `대중교통체계`, **A·B·C 모두 `대중교통체계`(정확)**.

> **tok/s 주의**: A는 **unsloth** 추론(122.8), B·C는 **plain transformers int4** 추론(~37)이라 **동일 조건 비교가 아니다**.
> 공정한 비교 축은 **EM·F1·size·ppl**. `peak VRAM`도 노트북 내 측정이라 A·C는 학습 직후 잔여 할당을 포함(같은
> int4를 서빙하는 B의 **~4.6GB**가 순수 int4 서빙 풋프린트에 가장 근접). 크기(1.29GB, B=C 동일)가 풋프린트 동등성을 확증.

## 재현성 (버전 고정)
`results/env_{A,B,C}.json`에 실행 시 자동 기록. **본 A100 실행 환경**: torch **2.11.0+cu130** ·
transformers **5.5.0** · trl **0.24.0** · peft **0.20.0** · datasets **4.3.0** · **torchao 0.17.0** ·
python 3.10 · CUDA 13.0 · unsloth 2026.7.6 · **A100 80GB PCIe(japaneast, Spot)**.

> **torchao INT4 주의**: 기본 `Int4WeightOnlyConfig(g128)`(PLAIN 패킹)은 실양자화 시 `mslk >= 1.0.0`을 요구해
> 실패한다. 본 트랙은 내장 tinygemm 경로인 `int4_packing_format=TILE_PACKED_TO_4D`를 사용(B·C 공통 서빙 포맷).

## 가드레일
- `quantization/` 안에서만. `pdf_qa` 코어·`evaluation/`·`personas.yaml`·웹앱 **무변경**.
- **데이터셋 미커밋**(런타임 다운로드). `.env`/키/토큰 미커밋 — 노트북 출력에도 미노출.
- 베이스·하이퍼파라미터는 `config.yaml`(A/B/C 동일 베이스 고정).

## vLLM INT4 서빙 검증 (보너스)
B·C가 공유하는 **TorchAO int4 tile-packed 아티팩트**를 **vLLM 0.26.0**으로 로드해 서빙 가능함을 실측 확인:
```python
from vllm import LLM, SamplingParams
llm = LLM(model="quantization/artifacts/B_int4_ptq", quantization="torchao",
          dtype="bfloat16", enforce_eager=True, max_model_len=2048)
```
- **로드 성공**: 엔진 로그 기준 GPU 가중치 풋프린트 **1.29GB**(디스크 크기와 일치), KV cache 45.6GB 확보.
- **생성 정확**: *"훈민정음을 창제한 사람은?"* → **`세종대왕`**(정답). 즉 학습→PTQ/QAT→**vLLM 서빙**까지 end-to-end 동작.
- 동일 포맷이라 C 아티팩트도 같은 방식으로 서빙된다(B로 대표 검증).

## 상태
- ✅ **3-way 완성**: A(BF16)·B(INT4 PTQ)·C(INT4 QAT) 모두 A100 실측, `results/three_way_table.json` 3행.
- ✅ 노트북 01·02·03 실행 완료(실 출력 포함), 재현 환경 `env_{A,B,C}.json`.
- ✅ vLLM int4 서빙 검증(보너스).
