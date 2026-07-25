# QA 스코어러 리포트 — `qa_memoirist_v2_enlarged`

- 총 쌍: **10** (judge 판정: 10)
- **strict PASS**: 0/10 (0%)  — REGISTER∧FIRST_PERSON∧GROUNDED∧COHERENT∧VOICE_PRESERVED∧Q_GROUNDED
- **lenient PASS**: 0/10 (0%)  — GROUNDED∧COHERENT∧REGISTER

## 차원별 실패 수

| 차원 | 실패 | 경고 |
|---|---|---|
| FORMAT | 0 | 0 |
| REGISTER(존댓말) | 10 | 0 |
| FIRST_PERSON | 0 | 6 |
| GROUNDED(날조) | 1 | 0 |
| COHERENT(비문/왜곡) | 0 | 0 |
| VOICE_PRESERVED | 1 | 0 |
| Q_GROUNDED | 0 | 0 |
| LEADING_Q(유도질문) | 0 | 1 |

## 청크 축 집계 (존댓말 lock 확인용)

| chunk | 쌍 | strict | REGISTER 실패 |
|---|---|---|---|
| 0 | 5 | 0/5 | 5 |
| 1 | 5 | 0/5 | 5 |

## 쌍별 판정

| # | strict | 실패 차원 | 답변(앞 40자) |
|---|---|---|---|
| 0 | ❌ | register | 열다섯 나던 해에 난리가 나서 온 식구가 봇짐을 이고 피난을 갔습니다.  |
| 1 | ❌ | register | 도회지로 올라와 낮에는 지게를 지고 밤에는 야학에 다녔습니다. 남들이 잠 |
| 2 | ❌ | register | 사흘 밤낮으로 비가 퍼붓고 강물이 넘쳐 집이 반이나 물에 잠겼습니다. 세 |
| 3 | ❌ | register | 안사람이 십 년을 병으로 누웠을 때, 손수 미음을 쑤어 떠 넣어 주며 젊 |
| 4 | ❌ | register | 해방이 되던 날, 온 동네가 만세를 부르며 거리로 쏟아져 나왔습니다. 저 |
| 5 | ❌ | register, grounded, voice_preserved | 그때는 검정이 되었던 시기였고, 이웃들이 번갈아 가며 양식을 나누어 주었 |
| 6 | ❌ | register | 환갑날 자식들이 없는 돈을 모아 조촐한 잔치를 차려 주었고, 색동옷을 입 |
| 7 | ❌ | register | 아무리 살림이 어려워도 조상 앞에 메 한 그릇, 탕 한 그릇은 정성껏 올 |
| 8 | ❌ | register | 안사람을 먼저 보내고 나서 한동안 넋을 놓고 지냈습니다. 밥상을 마주하여 |
| 9 | ❌ | register | 손주가 할아버지는 왜 이렇게 이상한 글자를 쓰느냐고 물었을 때, 허허 웃 |
