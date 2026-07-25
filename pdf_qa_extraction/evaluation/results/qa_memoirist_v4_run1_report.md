# QA 스코어러 리포트 — `qa_memoirist_v4_run1`

- 총 쌍: **10** (judge 판정: 10)
- **strict PASS**: 4/10 (40%)  — REGISTER∧FIRST_PERSON∧GROUNDED∧COHERENT∧VOICE_PRESERVED∧Q_GROUNDED
- **lenient PASS**: 4/10 (40%)  — GROUNDED∧COHERENT∧REGISTER

## 차원별 실패 수

| 차원 | 실패 | 경고 |
|---|---|---|
| FORMAT | 0 | 0 |
| REGISTER(존댓말) | 5 | 0 |
| FIRST_PERSON | 0 | 6 |
| GROUNDED(날조) | 1 | 0 |
| COHERENT(비문/왜곡) | 1 | 0 |
| VOICE_PRESERVED | 1 | 0 |
| Q_GROUNDED | 0 | 0 |
| LEADING_Q(유도질문) | 0 | 1 |

## 청크 축 집계 (존댓말 lock 확인용)

| chunk | 쌍 | strict | REGISTER 실패 |
|---|---|---|---|
| 0 | 5 | 0/5 | 5 |
| 1 | 5 | 4/5 | 0 |

## 쌍별 판정

| # | strict | 실패 차원 | 답변(앞 40자) |
|---|---|---|---|
| 0 | ❌ | register | 어린 시절 서당에 다니며 글을 배울 때 가장 기억에 남는 훈장님의 말씀은 |
| 1 | ❌ | register | 난리 중 어린 누이를 잃었던 일은 지금도 가슴이 미어지는 듯하며, 실로  |
| 2 | ❌ | register | 안사람이 병으로 누워 계실 때, 젊은 시절 다정하지 못했던 것을 뉘우치며 |
| 3 | ❌ | register | 해방이 되던 날, 온 동네가 만세를 부르며 거리로 나왔고, 목이 터지도록 |
| 4 | ❌ | register | 젊은이들에게 부모를 섬기기를 다하라고 한 이유는 부모가 살아 계실 때 잘 |
| 5 | ❌ | grounded, coherent, voice_preserved | 그때 나는 검정이 되었던 상황이었고, 이웃들이 번갈아 가며 양식을 나누어 |
| 6 | ✅ | — | 환갑날 자식들이 없는 돈을 모아 조촐한 잔치를 차려 주었고, 나는 색동옷 |
| 7 | ✅ | — | 나는 아무리 살림이 어려워도 조상 앞에 메 한 그릇, 탕 한 그릇은 정성 |
| 8 | ✅ | — | 안사람을 먼저 보내고 나서 한동안 넋을 놓고 지냈으며, 밥상을 마주하여도 |
| 9 | ✅ | — | 손주가 할아버지는 왜 이렇게 이상한 글자를 쓰느냐고 물었을 때, 나는 허 |
