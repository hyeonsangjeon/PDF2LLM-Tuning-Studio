# PDF-native benchmark — 6-arm results (historical_final/v1)

- base model: `Qwen/Qwen3-8B`  · seeds: [42, 43, 44]  · retriever: bm25 k=4 (`4b1db1efbde5`)

## Aggregate (mean ± std over seeds)

| arm | EM | F1 | cite-span | grounded | PII-leak | size GB |
|---|---|---|---|---|---|---|
| base_bf16 | 0.0±0.0 | 0.2146±0.0 | 0.0±0.0 | 1.0±0.0 | 0.0±0.0 | not_measured |
| sft_bf16 | 0.0±0.0 | 0.2176±0.0036 | 0.0±0.0 | 0.9722±0.0 | 0.0±0.0 | 15.2713±0.0 |
| sft_int4_ptq | 0.0±0.0 | 0.2222±0.0032 | 0.0±0.0 | 0.9537±0.0131 | 0.0±0.0 | 5.7705±0.0 |
| sft_int4_qat | 0.0±0.0 | 0.2146±0.0 | 0.0±0.0 | 1.0±0.0 | 0.0±0.0 | 5.7705±0.0 |
| base_bf16_retrieval | 0.0±0.0 | 0.3332±0.0 | 0.4516±0.0 | 0.9167±0.0 | 0.0±0.0 | not_measured |
| sft_bf16_retrieval | 0.2258±0.0 | 0.4442±0.0011 | 0.4516±0.0 | 0.8611±0.0 | 0.0±0.0 | 15.2713±0.0 |

## Spec comparisons (paired bootstrap, 95% CI)

- **sft_vs_base_closed_book** (sft_bf16 − base_bf16): ΔF1 = 0.0081 CI95 [0.0, 0.0242] (n=31, significant=False)
- **retrieval_effect_on_base** (base_bf16_retrieval − base_bf16): ΔF1 = 0.1186 CI95 [0.0514, 0.1983] (n=31, significant=True)
- **sft_retrieval_vs_base_retrieval** (sft_bf16_retrieval − base_bf16_retrieval): ΔF1 = 0.1117 CI95 [0.0558, 0.1804] (n=31, significant=True)

## Wrong examples (arm `base_bf16@42`) — reason visible

- `q000` [wrong_answer,numeric_unit,citation,over_abstention] Q=2024년 연간 매출액은 얼마입니까? gold=1,250억 원입니다. pred=문서에서 확인할 수 없습니다.
- `q001` [wrong_answer,numeric_unit,citation,over_abstention] Q=전년 대비 매출 성장률은 몇 퍼센트입니까? gold=12.5% 증가하였습니다. pred=문서에서 확인할 수 없습니다.
- `q002` [wrong_answer,numeric_unit,citation,over_abstention] Q=영업이익은 얼마입니까? gold=320억 원입니다. pred=문서에서 확인할 수 없습니다.
- `q003` [wrong_answer,numeric_unit,citation,over_abstention] Q=영업이익률은 몇 퍼센트입니까? gold=25.6%입니다. pred=문서에서 확인할 수 없습니다.
- `q004` [wrong_answer,numeric_unit,citation,over_abstention] Q=당기순이익은 얼마입니까? gold=210억 원입니다. pred=문서에서 확인할 수 없습니다.
- `q005` [wrong_answer,citation,over_abstention] Q=대표이사는 누구입니까? gold=홍길동입니다. pred=문서에서 확인할 수 없습니다.
- `q006` [wrong_answer,citation,over_abstention] Q=본사는 어디에 위치합니까? gold=가상시 합성구에 위치합니다. pred=문서에서 확인할 수 없습니다.
- `q007` [wrong_answer,numeric_unit,citation,over_abstention] Q=1분기 매출액은 얼마입니까? gold=280억 원입니다. pred=문서에서 확인할 수 없습니다.

## Honesty

- Aggregate is regenerated from per_example/*.jsonl (never hand-typed).
- Public frozen regression fixture (small N); confidence intervals are wide by design.
- Closed-book arms receive NO document context -> low answerability is expected and honest.
- 'fine-tuning effect' is only claimed if the base arm is present AND the paired CI excludes 0.
