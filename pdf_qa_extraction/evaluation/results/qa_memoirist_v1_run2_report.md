# QA 스코어러 리포트 — `qa_memoirist_v1_run2`

- 총 쌍: **10** (judge 판정: 10)
- **strict PASS**: 5/10 (50%)  — REGISTER∧FIRST_PERSON∧GROUNDED∧COHERENT∧VOICE_PRESERVED∧Q_GROUNDED
- **lenient PASS**: 5/10 (50%)  — GROUNDED∧COHERENT∧REGISTER

## 차원별 실패 수

| 차원 | 실패 | 경고 |
|---|---|---|
| FORMAT | 0 | 0 |
| REGISTER(존댓말) | 5 | 0 |
| FIRST_PERSON | 0 | 2 |
| GROUNDED(날조) | 1 | 0 |
| COHERENT(비문/왜곡) | 0 | 0 |
| VOICE_PRESERVED | 0 | 0 |
| Q_GROUNDED | 1 | 0 |
| LEADING_Q(유도질문) | 0 | 2 |

## 청크 축 집계 (존댓말 lock 확인용)

| chunk | 쌍 | strict | REGISTER 실패 |
|---|---|---|---|
| 0 | 5 | 0/5 | 5 |
| 1 | 5 | 5/5 | 0 |

## 쌍별 판정

| # | strict | 실패 차원 | 답변(앞 40자) |
|---|---|---|---|
| 0 | ❌ | register | 나는 아홉 살 되던 해에 처음으로 서당에 나갔습니다. 십 리 길을 매일  |
| 1 | ❌ | register | 열다섯 되던 해에 난리가 나서 온 식구가 봇짐을 이고 피난을 갔습니다.  |
| 2 | ❌ | register, grounded, q_grounded | 난리가 끝난 뒤 무작정 도회지로 올라와 낮에는 지게를 지고 밤에는 야학에 |
| 3 | ❌ | register | 병술년 여름에 큰물이 나서 사흘 밤낮으로 비가 퍼붓더니 강물이 넘쳐 집이 |
| 4 | ❌ | register | 해방이 되던 날 온 동네가 만세를 부르며 거리로 쏟아져 나왔습니다. 나도 |
| 5 | ✅ | — | 검정이 되었을 때 이웃들이 번갈아 양식을 나누어 준 그 고마움을 나는 평 |
| 6 | ✅ | — | 환갑날 자식들이 없는 돈을 모아 조촐한 잔치를 차려 주었을 때, 나는 색 |
| 7 | ✅ | — | 나는 제삿날을 한 번도 거르지 아니하였다. 아무리 살림이 어려워도 조상  |
| 8 | ✅ | — | 안사람을 먼저 보내고 나서, 나는 한동안 넋을 놓고 지냈다. 밥상을 마주 |
| 9 | ✅ | — | 이제 내 나이 여든 하고도 다섯이라, 언제 눈을 감아도 여한이 없다. 다 |
