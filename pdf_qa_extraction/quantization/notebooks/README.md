# notebooks/ — 튜닝 방법별 설명형 노트북 (공통 템플릿 · **v2 전면 개편**)

각 방법(A/B/C)마다 **설명 + 실행 + 데모 + 수치**를 담은 자기문서화 노트북. A100 80GB에서 실제
실행한 **실 출력을 커밋**한다(빈 출력 금지, 키·토큰 미노출).

> **v2**: base=**Qwen3-8B**, transformers+peft+trl(HF 백엔드), LoRA 2 epoch(750 step), TorchAO int4,
> **3 seed(42·43·44)**. 노트북은 재학습이 아니라 **실 아티팩트·집계표를 로드**하고 held-out 데모 1건만
> 실행하므로(수치 비의존) 재현이 가볍고 결정적이다.

| 노트북 | 내용 | 상태 |
|---|---|---|
| `00_base_select.ipynb` | 베이스 ≤9B 3종 zero/few-shot F1 → 선정(Qwen3-8B) | **A100 실측(실 출력)** |
| `01_bf16_lora.ipynb` | **Method A** BF16 LoRA 설명+로드+데모+수치 | **A100 실측(실 출력)** |
| `02_int4_ptq.ipynb` | **Method B** INT4 PTQ (TorchAO tile-packed) | **A100 실측(실 출력)** |
| `03_int4_qat.ipynb` | **Method C** INT4 QAT (full-param STE) | **A100 실측(실 출력)** |

## 베이스 선정 harness — `bench_baseselect.py` (재현 가능 CLI)

`00_base_select.ipynb`가 참조하는 선정 스크립트를 실행 가능한 CLI로 복원했다(정식 chat template ·
`enable_thinking=false` · 적정 `max_new_tokens` · 공식 KorQuAD char-level EM/F1 — v1의 W1 harness 아티팩트
수정). 후보·템플릿·token budget·few-shot·seed·eval slice는 모두 `config.yaml`에서 읽는다.

```bash
cd pdf_qa_extraction
# (1) 후보 해석 계획만 출력(모델 미로드): gated Llama → ungated Yi 폴백 표시
python -m quantization.bench_baseselect --list-candidates
# (2) 커밋된 결과(results/base_select.json) 읽기 전용 점검: SHA·정렬·winner (파일 미변경)
python -m quantization.bench_baseselect --check-historical
# (3) CPU smoke — 작은 ungated 모델로 전 경로 실행(GPU 불필요). 출력은 runs/ 아래
python -m quantization.bench_baseselect --smoke --run-id demo
# (4) 실제 GPU 선정 재실행(A100): 결과는 runs/<run_id>/quantization/base_select.json
python -m quantization.bench_baseselect --fewshot 2
```

- 커밋된 `quantization/results/base_select.json`은 **A100 VM 실측 historical 아티팩트**이며, CLI는 이를
  **읽기 전용**으로 다루고 fresh run은 `runs/` 아래에만 쓴다(historical SHA-256 불변 보장). `--check-historical`은
  이 파일을 새 코드가 생성한 것처럼 소급 주장하지 않고 `status=historical_not_reproduced`로 표시한다.
- 스키마·후보 정렬·seed 고정·읽기전용 계약은 `quantization/tests/test_base_select.py`로 검증된다.

## 공통 5-파트 템플릿
1. **설명(markdown)** — 이 방법이 무엇/왜/어떻게 다른가 (A=BF16 기준선, B=PTQ 사후, C=QAT 인식학습).
2. **부트스트랩+설정** — `pdf_qa_extraction`를 import 경로에 넣고 repo 루트로 `chdir`(실행 위치 무관), `config.yaml` 로드.
3. **산출물 로드** — `artifacts/{method}_seed42` + `results/`의 실 metrics/집계표를 로드(코드 중복 없이 얇게 오케스트레이션).
4. **동작 데모(필수)** — 튜닝 모델 로드 → **held-out KorQuAD 질문 1개** 생성 답변을 **실행 출력**으로 포함
   (`v2_pipeline.build_chat_prompt` + `extract_answer` + `model.generate`).
5. **수치 셀** — `results/three_way_table.json`(3 seed mean±std) + 방법별 eval json을 로드해 EM/F1·ppl·크기·VRAM·tok/s 표시.

## 실행 방법 (v2)
```bash
cd pdf_qa_extraction
pip install nbconvert ipykernel
python -m ipykernel install --user --name python3
# FlashInfer 샘플러 JIT(nvcc) 회피 — 데모 셀의 vLLM/생성 안정화
export VLLM_USE_FLASHINFER_SAMPLER=0
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=python3 \
  quantization/notebooks/0{0,1,2,3}_*.ipynb
```
- 부트스트랩 셀이 `pdf_qa_extraction`를 경로에 넣고 repo 루트로 `chdir`하므로 실행 위치와 무관하게 동작한다.
- 노트북은 **재학습하지 않는다**(사전 산출물 로드 + 데모 1건만 생성) → 각 노트북 실행은 모델 로드 + 짧은 생성뿐.
  int4 노트북(02·03)은 데모 생성이 eager라 느릴 수 있다(수 초). 사전 산출물이 없으면 `quantization/v2_run`으로 먼저 생성한다.
- 상세는 각 노트북 상단 설명 셀 및 `../README.md`(컴퓨트/결과/벤치) 참고.
