"""5. Beam Search (NumPy).

매 스텝: 살아있는 빔마다 다음 토큰 log-prob을 더해 B*V 후보 → 상위 B개만 남김.
- 확률을 곱하지 말고 log-prob을 더한다 (underflow 방지)
- EOS 나온 빔은 finished로 빼고, 남은 자리로 계속
- 길이 패널티: score / len^alpha  (안 하면 짧은 문장 편애)
"""
import numpy as np


def beam_search(step_logprobs, bos, eos, beam_width=3, max_len=10, alpha=0.0):
    """step_logprobs(prefix: tuple[int]) -> np.ndarray (V,) log-prob.
    반환: [(정규화 점수, 원래 log-prob 합, 시퀀스)] 점수 내림차순."""
    beams = [(0.0, (bos,))]
    finished = []
    for _ in range(max_len):
        cand = []
        for lp, seq in beams:
            logp = step_logprobs(seq)
            # 빔당 2B개: 상위 후보 중 일부가 EOS로 빠져도 살아있는 빔 B개를 채울 수 있게 (HF도 2B)
            top = np.argsort(logp)[::-1][:2 * beam_width]
            for tok in top:
                cand.append((lp + float(logp[tok]), seq + (int(tok),)))
        cand.sort(key=lambda c: c[0], reverse=True)
        beams = []
        for lp, seq in cand:
            if seq[-1] == eos:
                finished.append((lp, seq))
            else:
                beams.append((lp, seq))
            if len(beams) == beam_width:
                break
        if not beams or len(finished) >= beam_width and max(f[0] for f in finished) > beams[0][0]:
            break   # 남은 빔은 log-prob이 더 내려갈 일만 있음 → 조기 종료 (alpha=0일 때 정확)
    finished += [b for b in beams if len(b[1]) - 1 >= max_len]   # max_len까지 간 미완료 빔만 후보로
    def norm(lp, seq):
        return lp / (max(len(seq) - 1, 1) ** alpha)
    return sorted(((norm(lp, s), lp, s) for lp, s in finished), reverse=True)


def greedy(step_logprobs, bos, eos, max_len=10):
    seq, lp = (bos,), 0.0
    for _ in range(max_len):
        logp = step_logprobs(seq)
        tok = int(np.argmax(logp)); lp += float(logp[tok]); seq += (tok,)
        if tok == eos:
            break
    return lp, seq


def make_toy_lm(V=6, seed=0, eos=0, peaky=1.5, min_len=3):
    """접두사에 따라 결정되는 무작위 LM (같은 prefix → 같은 분포). min_len 전엔 EOS 확률을 크게 낮춤."""
    rng = np.random.default_rng(seed)
    table = rng.normal(0, peaky, (V, V))
    def f(seq):
        h = (hash(seq) + seed * 1_000_003) % (2 ** 32)
        noise = np.random.default_rng(h).normal(0, 1.0, V)
        z = table[seq[-1]] + noise
        if len(seq) < min_len:
            z = z.copy(); z[eos] -= 6.0
        return z - np.log(np.exp(z - z.max()).sum()) - z.max()
    return f


def classic_example():
    """교과서 예: greedy는 첫 스텝 0.6을 고르지만, 0.4 쪽이 뒤에서 0.9라 전체가 더 높다.
    토큰: 0=EOS 1=BOS 2=A 3=B 4=C"""
    P = {(1,): {2: .6, 3: .4},
         (1, 2): {4: .4, 0: .3, 3: .3}, (1, 3): {4: .9, 0: .05, 2: .05},
         (1, 2, 4): {0: 1.0}, (1, 3, 4): {0: 1.0}, (1, 2, 3): {0: 1.0}, (1, 3, 2): {0: 1.0}}
    def f(seq):
        lp = np.full(5, -1e9)
        for t, p in P.get(tuple(seq), {0: 1.0}).items():
            lp[t] = np.log(p)
        return lp
    return f


if __name__ == "__main__":
    ex = classic_example()
    print("classic greedy:", greedy(ex, 1, 0), " beam:", beam_search(ex, 1, 0, beam_width=2)[0][1:])
    lm = make_toy_lm()
    print("greedy:", greedy(lm, bos=1, eos=0))
    for s, lp, seq in beam_search(lm, bos=1, eos=0, beam_width=3)[:3]:
        print(f"beam  : logp={lp:.3f} {seq}")
