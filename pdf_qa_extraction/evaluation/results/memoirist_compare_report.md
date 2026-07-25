# QA 스코어러 비교 리포트 — `memoirist`

변이별 재판정 결과(스코어러). 수기 v1-vs-v4 집계를 대체.

| 변이 | 런 | 총쌍 | strict PASS | lenient PASS | REGISTER실패 | GROUNDED실패 | COHERENT실패 | VOICE실패 |
|---|---|---|---|---|---|---|---|---|
| **v4** | 3 | 30 | 22/30 (73%); run min/max/mean 4/9/7.3 | 22/30 (73%) | 5 | 3 | 3 | 3 |
| **v1** | 3 | 30 | 24/30 (80%); run min/max/mean 5/10/8.0 | 25/30 (83%) | 5 | 1 | 0 | 1 |
| **v2** | 1 | 10 | 0/10 (0%); run min/max/mean 0/0/0.0 | 0/10 (0%) | 10 | 1 | 0 | 1 |

## 런별 상세

### v4

| 런 | 총쌍 | strict | lenient | REG실패 | GROUND실패 | COHER실패 | VOICE실패 |
|---|---|---|---|---|---|---|---|
| qa_memoirist_v4_run1 | 10 | 4 | 4 | 5 | 1 | 1 | 1 |
| qa_memoirist_v4_run2 | 10 | 9 | 9 | 0 | 1 | 1 | 1 |
| qa_memoirist_v4_run3 | 10 | 9 | 9 | 0 | 1 | 1 | 1 |

### v1

| 런 | 총쌍 | strict | lenient | REG실패 | GROUND실패 | COHER실패 | VOICE실패 |
|---|---|---|---|---|---|---|---|
| qa_memoirist_v1_run1 | 10 | 10 | 10 | 0 | 0 | 0 | 0 |
| qa_memoirist_v1_run2 | 10 | 5 | 5 | 5 | 1 | 0 | 0 |
| qa_memoirist_v1_run3 | 10 | 9 | 10 | 0 | 0 | 0 | 1 |

### v2

| 런 | 총쌍 | strict | lenient | REG실패 | GROUND실패 | COHER실패 | VOICE실패 |
|---|---|---|---|---|---|---|---|
| qa_memoirist_v2_enlarged | 10 | 0 | 0 | 10 | 1 | 0 | 1 |

