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

## ⚙️ 컴퓨트 / GPU 쿼터 상태 (중요)
스펙은 **Azure 단일 GPU VM** 실행을 요구한다. 확인 결과, 대상 구독
`ME-MngEnvMCAP756842-hjeon-1`에는 **모던 GPU 패밀리 쿼터가 전부 0**이었다
(H100/A100/A10/T4 모두 limit=0; 리테이어드 K80·M60만 non-zero라 BF16/unsloth 불가).

지시대로 **쿼터 증설을 요청**했으나 승인되지 않았다:

| 패밀리 | 요청 vCPU | 요청 ID | 결과 |
|---|---|---|---|
| H100 `StandardNCadsH100v5Family` | 40 | `e9245f08` | **Failed** |
| A10 `StandardNVADSA10v5Family` | 36 | `d9addbec` | **Failed** |
| A100 `StandardNCADSA100v4Family` | 24 | `bd193591` | InProgress→Failed |

과거 동일 패밀리 요청(H100→160, A100→96, T4→16)도 **모두 Failed**, `CORES`만 Succeeded
→ 스폰서(MCAP) 구독의 **GPU 쿼터 잠금**으로 판단(코드/권한 문제 아님).

**따라서 "쿼터 중 있는 걸로 처리"** 지시에 맞춰, 파이프라인을 **동일 코드 경로**로
로컬 **CPU 스모크**(소형 `Qwen/Qwen2.5-0.5B-Instruct` + 200 서브셋 + 12스텝)로 실제 실행하여
모든 블록(데이터·지표·학습·머지·데모·수치)이 통과함을 **실 출력**으로 노트북에 커밋했다.
GPU 쿼터가 열리면 `compute.mode: gpu`로 **동일 셀**을 돌려 스펙의 A100/H100 BF16 수치를 채우면 된다.

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

## Method A 결과 — 3-way 표 첫 행
> ⚠️ 아래는 **CPU 스모크**(0.5B·12스텝·fp32) 수치로 **파이프라인 정합성 증거**다.
> 스펙 수치(Qwen3-1.7B·BF16·full·A100/H100)는 GPU에서 `mode=gpu` 재실행 시 채워진다.

| method | base | EM | F1 | ppl | size(GB) | VRAM | tok/s | prec |
|---|---|---|---|---|---|---|---|---|
| A_bf16 (smoke) | Qwen2.5-0.5B | 30.0 | 45.2 | 15.7 | 1.85 | n/a(CPU) | 5.6 | fp32 |

참고: 동일 모델 **zero-shot F1 18.0 → LoRA 후 45.2** (학습이 실제로 개선; 00 vs 01 노트북).

## 재현성 (버전 고정)
`results/env_A.json`에 실행 시 자동 기록: torch 2.13.0 · transformers 5.14.1 · trl 1.9.0 ·
peft 0.19.1 · datasets 5.0.0 · python 3.10. GPU VM에선 CUDA/unsloth/torchao/vllm 버전을
동일하게 기록한다.

## 가드레일
- `quantization/` 안에서만. `pdf_qa` 코어·`evaluation/`·`personas.yaml`·웹앱 **무변경**.
- **데이터셋 미커밋**(런타임 다운로드). `.env`/키/토큰 미커밋 — 노트북 출력에도 미노출.
- 베이스·하이퍼파라미터는 `config.yaml`(A/B/C 동일 베이스 고정).

## 다음 (Part 2)
`artifacts/A_bf16/` 머지 모델 → **INT4 PTQ(B)** / **INT4 QAT(C)** + 노트북 02·03 → 3-way 표 완성 → vLLM 서빙 벤치.
