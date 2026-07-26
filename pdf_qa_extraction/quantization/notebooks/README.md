# notebooks/ — 튜닝 방법별 설명형 노트북 (공통 템플릿)

각 방법(A/B/C)마다 **설명 + 실행 + 데모 + 수치**를 담은 자기문서화 노트북. VM에서 실제
실행한 **실 출력을 커밋**한다(빈 출력 금지, 키·토큰 미노출).

| 노트북 | 내용 | 상태 |
|---|---|---|
| `00_base_select.ipynb` | 베이스 3종 zero-shot F1 → 선정 | 실행됨(CPU 스모크 프록시 1종 실측) |
| `01_bf16_lora.ipynb` | **Method A** 설명+학습+데모+수치 | **실행됨(실 출력 커밋)** |
| `02_int4_ptq.ipynb` | Method B (PTQ) | 템플릿 스텁(Part 2) |
| `03_int4_qat.ipynb` | Method C (QAT) | 템플릿 스텁(Part 2) |

## 공통 5-파트 템플릿
1. **설명(markdown)** — 이 방법이 무엇/왜/어떻게 다른가 (A=BF16 기준선, B=PTQ 사후, C=QAT 인식학습).
2. **코드** — `quantization/` 공용 모듈을 import해 **얇게 오케스트레이션**(코드 중복 금지; 노트북=서사+실행).
3. **실행 셀** — 학습/양자화 (또는 사전 산출물 로드).
4. **동작 데모(필수)** — 튜닝 모델 로드 → **held-out KorQuAD 질문 1개** 생성 답변을 **실행 출력**으로 포함.
5. **수치 셀** — EM/F1·ppl·크기·VRAM·tok/s → `results/` + 3-way 표 append.

## 실행 방법
```bash
cd pdf_qa_extraction
pip install nbconvert ipykernel
python -m ipykernel install --user --name python3
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
  quantization/notebooks/01_bf16_lora.ipynb
```
- 첫 코드 셀이 `quantization` 패키지를 import 경로에 넣고 repo 루트로 `chdir`하므로,
  실행 위치에 관계없이 동작한다.
- **GPU VM**: `load_config()`(= `compute.mode: gpu`, Qwen3-1.7B·unsloth·BF16·full).
  **무 GPU**: 노트북은 `load_config(force_mode='cpu')` 스모크로 실행됨(소형 모델·서브셋).
  상세는 각 노트북 상단 **실행 모드 배너** 및 `../README.md` 컴퓨트/쿼터 상태 참고.
