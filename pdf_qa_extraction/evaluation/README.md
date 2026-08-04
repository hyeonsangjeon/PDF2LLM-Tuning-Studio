# memoirist QA 스코어러 (`evaluation/`)

사람이 눈으로 하던 **문체 보존 / 날조 / 존댓말 / 왜곡** 판정을 반복 가능한 자동
스코어러로 만든 모듈입니다. 두 용도:

1. **데이터셋 QC 게이트** — 학습에 넣기 전 불량 QA 쌍을 걸러/버림.
2. **회귀 eval** — 페르소나 버전(v1/v4/…)·추출 모델을 객관 비교(수기 PASS/FAIL 집계 대체).

> 격리: `pdf_qa` 코어 · `personas.yaml` · 웹앱은 **건드리지 않습니다**. 이 모듈은
> `pdf_qa`의 순수 헬퍼(`custom_json_parser`, PDF 파서)만 *읽어서* 재사용합니다.

## 2층 구조 (규칙은 전부 `rubric.yaml`, 코드 하드코딩 없음)

| 층 | 비용 | 검출 |
|---|---|---|
| **Layer 1 — 결정론(LLM 없음)** | 즉시·전 쌍 | `REGISTER`(존댓말 종결) · `FIRST_PERSON` · `LEADING_Q`(유도질문) · `FORMAT`(빈/중복/스키마) |
| **Layer 2 — LLM judge(temp=0)** | 쌍당 1콜 | `GROUNDED`(날조) · `COHERENT`(비문/왜곡압축) · `VOICE_PRESERVED`(옛 문어체 유지, 철자현대화는 허용) · `Q_GROUNDED` |

- **REGISTER**는 답변 종결어미를 한글 자모(ㅂ받침) 기준으로 판정하므로
  `입니다/합니다/생생합니다`(존댓말)는 잡고 `아니다/지니다`(문어체)는 통과합니다.
- **judge는 생성기와 독립 호출**입니다. 이 데이터는 gpt-4o 생성이므로 판정 정합성
  검증 후 judge도 gpt-4o(temp=0)를 채택했습니다(캘리브레이션 근거는 아래 참고).

## PASS 정의
- **strict PASS** = `REGISTER ∧ FIRST_PERSON ∧ GROUNDED ∧ COHERENT ∧ VOICE_PRESERVED ∧ Q_GROUNDED`
- **lenient PASS** = `GROUNDED ∧ COHERENT ∧ REGISTER` (핵심 3개)
- 리포트에 **둘 다** 병기.

## 사용법

```bash
# 단일 데이터셋 QC (Azure judge, temp=0). 소스는 pdf/txt/md 모두 가능.
python -m evaluation.run_eval score \
    --qa data/samples/qa_memoirist_v4_run1.jsonl \
    --source data/samples/memoir_sample_ko_long.pdf \
    --judge-provider azure --judge-model gpt-4o --pairs-per-chunk 5

# 자격증명 없이 Layer 1(결정론)만 — 존댓말/형식 게이트만 빠르게
python -m evaluation.run_eval score --qa run.jsonl --source src.txt --no-judge

# 변이 비교(수기 v1-vs-v4 집계 대체)
python -m evaluation.run_eval compare --source data/samples/memoir_sample_ko_long.pdf \
    --judge-model gpt-4o --pairs-per-chunk 5 --name memoirist \
    --variant v4=data/samples/qa_memoirist_v4_run1.jsonl,data/samples/qa_memoirist_v4_run2.jsonl,data/samples/qa_memoirist_v4_run3.jsonl \
    --variant v1=data/samples/qa_memoirist_v1_run1.jsonl,data/samples/qa_memoirist_v1_run2.jsonl,data/samples/qa_memoirist_v1_run3.jsonl \
    --variant v2=data/samples/qa_memoirist_v2_enlarged.jsonl
```

### 출력 (`results/`)
- `<name>_scored.jsonl` — 각 쌍 + 차원별 판정 + 근거
- `<name>_clean.jsonl` — strict PASS만(학습 후보)
- `<name>_rejected.jsonl` — 탈락 쌍 + 사유(재생성용)
- `<name>_report.md` — strict/lenient PASS율, 차원별 실패 수, **run×chunk**(존댓말 lock 확인)
- `<name>_compare_report.md` — 변이 비교표(run별 + min/max/mean)

## 메타-eval (스코어러 자체 검증, `tests/`)
judge를 신뢰하기 전에 **사람이 라벨한 골든 케이스로 캘리브레이션**합니다("판정자를 판정").

- **Layer 1**은 커밋된 샘플에 직접 어서션(결정론) — v4_run1 앞 5쌍 REGISTER FAIL,
  v2 10/10 REGISTER FAIL, v1·v4 각 5/30 존댓말(수기 수치 재현).
- **Layer 2**는 `tests/fixtures/judge_cache_gpt4o.json`(실제 gpt-4o judge를 1회
  기록)을 `ReplayJudge`로 재생 → **자격증명·네트워크 없이** 골든을 검증.
  - 검정("숯검정이 되었다"→"검정이 되었던 상황") 쌍 → COHERENT FAIL
  - v4_run2/run3의 plain·grounded 쌍 → strict PASS
  - v4 30쌍 strict ≈ 22/30(= 30 − 존댓말 5 − 검정왜곡 3) — 스펙 목표(20~22) 적중

```bash
python -m pytest evaluation/tests/ -q      # 자격증명 불필요
```

## Programmatic-verifier reward interface (RL은 `planned`)

`rewards.py`는 향후 가능한 RL 단계를 위해 **결정적(deterministic) reward 함수 +
RewardCard**를 *먼저* 정의합니다 — evidence 무결성·수치 정합·schema 준수·규칙 기반
계산·abstention·PII 비노출·최신 버전 선택 + 길이/reward-hacking 가드. RL은 **실행되지
않았고**(`RL_STATUS="planned"`), GRPO/PPO는 타당성 게이트 통과 전까지 추가하지 않습니다.
설계·게이트·완료조건은 [`docs/RL_EXPERIMENT_PLAN.md`](../../docs/RL_EXPERIMENT_PLAN.md).

```bash
python -m pytest evaluation/tests/test_rewards.py -q   # reward 단위 테스트(공개/합성만)
```

## judge 캘리브레이션 노트
같은 골든에서 **다른 모델**(gpt-5.4-mini)을 judge로 시험한 결과, 검정 왜곡을
`coherent=True`로 **놓쳤습니다**(gpt-4o는 `coherent=False`로 사람 라벨과 일치).
즉 이 데이터에선 gpt-4o judge가 사람 판정에 더 정합적이라 채택했고, "생성자=판정자"
맹점은 grounded/voice 축을 함께 봐 완화합니다. rubric·judge 모델은 YAML/CLI로 교체
가능하므로, 더 나은 독립 judge가 확인되면 `--judge-model`만 바꾸면 됩니다.
