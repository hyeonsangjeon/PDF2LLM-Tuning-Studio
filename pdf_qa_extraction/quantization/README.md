# quantization/ — 양자화 트랙 Part 1 (BF16 LoRA 베이스라인 A)

발표(Serve Track: **LoRA → INT4 PTQ/QAT → vLLM**)용 **3-way 양자화 비교**의 기반.
PDF 추출·페르소나·스코어러와 **무관한 독립 트랙**으로, 표준 데이터셋 **KorQuAD**(`KorQuAD/squad_kor_v1`,
런타임 다운로드·미커밋)만 사용한다.

| 방법 | 설명 | 노트북 | 상태 |
|---|---|---|---|
| **A. BF16 LoRA** | 풀정밀 베이스에 LoRA 학습 후 머지 (품질 기준선) | `01_bf16_lora.ipynb` | **이 Part 1 완성** |
| B. INT4 PTQ | A 머지 모델을 사후 4bit 양자화 (TorchAO) | `02_int4_ptq.ipynb` | 템플릿 스텁 (Part 2) |
| C. INT4 QAT | 4bit 인식 학습 | `03_int4_qat.ipynb` | 템플릿 스텁 (Part 2) |

## 구성
```
quantization/
  config.yaml          # A/B/C 공용 설정 (base·data·lora·train·compute). 스모크 오버라이드 포함
  data_korquad.py      # KorQuAD 로드 + 생성 instruction 포맷 + 고정 seed 분리
  train_lora.py        # Method A: BF16 LoRA. backend=unsloth(GPU) | hf(CPU-capable)
  eval_qa.py           # A/B/C 공용 eval: KorQuAD 공식 EM/F1 + ppl + 크기/VRAM/tok·s
  notebooks/           # 00 base-select, 01 BF16 LoRA(실행됨), 02/03 스텁, README(템플릿)
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
후보 3종과 선정 기준. 최종 확정은 `notebooks/00_base_select.ipynb`(GPU에서 3종 zero-shot F1).

| 후보 | 파라미터 | 라이선스 | 게이팅 | 비고 |
|---|---|---|---|---|
| **Qwen/Qwen3-1.7B** (기본 선정) | ~1.7B | Apache-2.0 | 무 | 한국어 양호·초경량·단일GPU·INT4/vLLM 성숙 |
| Qwen/Qwen3-4B | ~4B | Apache-2.0 | 무 | 품질↑, 여전히 단일GPU |
| meta-llama/Llama-3.2-3B-Instruct | ~3B | Llama | **유(승인+토큰)** | 패밀리 다양성 |

- **기준**: (a) KorQuAD dev zero-shot F1, (b) 단일 GPU 적합, (c) TorchAO INT4 + vLLM 서빙 호환.
- **기본값 = Qwen3-1.7B**(Apache-2.0·무게이팅·경량). `config.yaml`의 `base_model.selected`에 고정 →
  A/B/C 동일 베이스. CPU 스모크에선 3종 실측 불가(4B는 CPU 과도, Llama는 gated)라
  00 노트북이 소형 프록시 1종으로 **하네스 동작만 실증**(zero-shot F1 측정 성공).

## Method A 결과 — 3-way 표 첫 행 (실측 · A100 80GB)
`notebooks/01_bf16_lora.ipynb`를 **A100 80GB에서 실제 실행**한 수치(held-out 500):

| method | base | EM | F1 | ppl | size(GB) | peak VRAM | tok/s | prec |
|---|---|---|---|---|---|---|---|---|
| **A_bf16** | Qwen3-1.7B | **81.0** | **89.92** | 10.39 | 3.22 | 7.93 | 122.8 | bf16 |

- 학습: unsloth `FastLanguageModel`, BF16(load_in_4bit=false), LoRA r16/α32(attn+MLP),
  effective batch 8, **2000 steps**(≈16k KorQuAD 예제, ~0.26 epoch), 1× A100 80GB **~33분**, mean loss 2.03.
- 동작 데모(노트북 셀 실측): held-out 질문 *"2004년 이명박이 서울시장 재직시절 전면적으로 개선한 것은?"*
  → 정답 `대중교통체계`, **모델 답 `대중교통체계`(정확)**.
- B(INT4 PTQ)·C(INT4 QAT) 행은 Part 2에서 이 머지 BF16 모델을 입력으로 채운다.

## 재현성 (버전 고정)
`results/env_A.json`에 실행 시 자동 기록. **본 A100 실행 환경**: torch **2.11.0+cu130** ·
transformers **5.5.0** · trl **0.24.0** · peft **0.20.0** · datasets **4.3.0** · python 3.10 ·
CUDA 13.0 · unsloth 2026.7.6 · **A100 80GB PCIe**. Part 2(INT4/vLLM) 실행 시 torchao/vllm 버전도 동일 기록한다.

## 가드레일
- `quantization/` 안에서만. `pdf_qa` 코어·`evaluation/`·`personas.yaml`·웹앱 **무변경**.
- **데이터셋 미커밋**(런타임 다운로드). `.env`/키/토큰 미커밋 — 노트북 출력에도 미노출.
- 베이스·하이퍼파라미터는 `config.yaml`(A/B/C 동일 베이스 고정).

## 다음 (Part 2)
`artifacts/A_bf16/` 머지 모델 → **INT4 PTQ(B)** / **INT4 QAT(C)** + 노트북 02·03 → 3-way 표 완성 → vLLM 서빙 벤치.
