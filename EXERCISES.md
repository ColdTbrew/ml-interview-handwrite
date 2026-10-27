# 연습 문제 · 꼬리 질문 · 자가 점검

## 연습 방법 (추천 순서)

1. **종이 / 화이트보드에 먼저** (트윗의 조언). 타이머 켜고 아래 목표 시간 안에.
2. `blank/<이름>.py`에 그대로 옮겨 적기 (고치지 말고!).
3. `IMPL=blank pytest -q -k <키워드>` 로 채점. 틀린 테스트 = 종이에서 놓친 함정.
4. 틀린 부분만 다시 종이에. 3일 뒤 한 번 더.

| 알고리즘 | 목표 시간 [추론] | 채점 키워드 |
|---|---|---|
| Softmax (+log_softmax, backward) | 5분 | `-k softmax` |
| Scaled dot-product + MHA | 15분 | `-k "sdpa or mha"` |
| K-Means (+k-means++) | 15분 | `-k kmeans` |
| Adam / AdamW | 10분 | `-k adam` |
| Beam Search | 20분 | `-k beam` |

목표 시간은 내 추정이다. 트윗에는 시간 기준이 없다.

---

## 1. Softmax
- [ ] 최댓값을 빼도 결과가 같은 이유를 식 한 줄로 설명할 수 있다
- [ ] `axis`, `keepdims=True`를 빠뜨리지 않았다
- [ ] log_softmax를 `log(softmax(x))`로 쓰면 왜 나쁜지 안다 (작은 확률이 0 → log 0 = -inf)
- [ ] 꼬리 질문: softmax + cross-entropy를 합친 그래디언트는? → `p - y`
- [ ] 꼬리 질문: 온도 T, label smoothing은 확률 분포를 어떻게 바꾸나?

## 2. Attention / Multi-Head
- [ ] 각 줄 옆에 shape를 적었다: `(B,T,D) → (B,H,T,Dh) → (B,H,T,T) → (B,T,D)`
- [ ] reshape 후 transpose 순서를 지켰다 (반대로 하면 **에러 없이** 틀린 값)
- [ ] `√d_k`로 나누는 이유를 분산으로 설명할 수 있다
- [ ] causal mask가 상삼각(k=1)이라는 걸 그림으로 그릴 수 있다
- [ ] 꼬리 질문: 시간·메모리 복잡도? → O(T²·D), scores가 O(B·H·T²)
- [ ] 꼬리 질문: KV cache는 무엇을 저장하고 왜 디코딩이 빨라지나?
- [ ] 꼬리 질문: MQA / GQA는 MHA에서 무엇을 공유하나? (K,V head 수를 줄임)
- [ ] 꼬리 질문: FlashAttention은 수식을 바꾸나? (아니다. 타일링 + online softmax로 메모리 접근만 줄임)

## 3. K-Means
- [ ] 거리 계산을 for문 없이 `‖x‖² - 2x·c + ‖c‖²`로 벡터화했다
- [ ] 빈 클러스터 처리가 있다
- [ ] 수렴 조건(중심 이동 또는 라벨 불변)이 있다
- [ ] 꼬리 질문: 왜 inertia가 단조 감소하나? 그런데 왜 전역 최적은 아닌가?
- [ ] 꼬리 질문: k는 어떻게 고르나? (elbow, silhouette — 둘 다 휴리스틱)
- [ ] 꼬리 질문: 한 반복의 복잡도? → O(N·K·d)
- [ ] 꼬리 질문: 비구형·크기가 다른 클러스터엔? (GMM/EM, DBSCAN)

## 4. Adam
- [ ] m, v를 파라미터마다 따로 0으로 초기화했다
- [ ] t를 1부터 세고 편향 보정에 썼다
- [ ] ε은 `sqrt(v̂)` **밖**에 더했다 (torch 기본과 같음)
- [ ] Adam의 L2와 AdamW의 decoupled weight decay 차이를 코드 한 줄로 보여줄 수 있다
- [ ] 꼬리 질문: 파라미터당 추가 메모리? → 2배 (m, v). 그래서 8-bit Adam, Adafactor가 나옴
- [ ] 꼬리 질문: warmup은 왜 쓰나? (초반 v 추정이 불안정 — RAdam 논문의 설명)

## 5. Beam Search
- [ ] log-prob을 더했다 (확률을 곱하지 않았다)
- [ ] EOS 나온 빔을 finished로 분리했다
- [ ] 빔당 후보를 2B개 봤다 (또는 왜 필요한지 설명할 수 있다)
- [ ] 길이 정규화 `score / len^α`를 넣을 수 있다
- [ ] 꼬리 질문: 복잡도? → 스텝당 O(B·V log(B·V)), heap 쓰면 더 줄일 수 있음
- [ ] 꼬리 질문: 빔을 키우면 항상 좋아지나? (번역에서 "beam search curse": 너무 큰 빔이 짧고 빈 문장을 고름)
- [ ] 꼬리 질문: LLM 채팅은 왜 top-p/temperature 샘플링을 주로 쓰나?

---

## 보너스: 목록에 없지만 자주 나오는 것

[추론] 트윗 목록에 없는 단골 손코딩. 다음 단계로 이 레포에 추가할 후보.

- Cross-entropy loss + 그래디언트 (`p - y`)
- LayerNorm / RMSNorm (forward + 왜 eps가 분모 안에)
- Top-k / Top-p(nucleus) 샘플링
- 2층 MLP forward + backward (NumPy, 체인 룰)
- RoPE (회전 위치 인코딩)
- KV cache 붙인 디코딩 루프
- BPE 토크나이저 merge 한 단계
- Logistic regression 경사하강
