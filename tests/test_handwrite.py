"""손코딩 구현을 PyTorch / scikit-learn / 완전탐색과 대조한다.
torch, sklearn이 없으면 해당 테스트는 건너뛴다 (NumPy만으로도 나머지는 돈다)."""
import importlib
import itertools
import os
import numpy as np
import pytest

PKG = os.environ.get("IMPL", "handwrite")        # IMPL=blank 이면 내가 쓴 연습본을 채점
_m = lambda name: importlib.import_module(f"{PKG}.{name}")
softmax, log_softmax, softmax_backward = (getattr(_m("softmax"), n) for n in ("softmax", "log_softmax", "softmax_backward"))
scaled_dot_product_attention, causal_mask, MultiHeadAttention = (getattr(_m("attention"), n) for n in ("scaled_dot_product_attention", "causal_mask", "MultiHeadAttention"))
kmeans = _m("kmeans").kmeans
Adam = _m("adam").Adam
beam_search, greedy = _m("beam_search").beam_search, _m("beam_search").greedy
from handwrite.beam_search import make_toy_lm, classic_example

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch 없음")
rng = np.random.default_rng(0)


# ---------- 1. softmax ----------
def test_softmax_stable_large_inputs():
    y = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.all(np.isfinite(y)) and np.isclose(y.sum(), 1)


def test_softmax_shift_invariance():
    x = rng.normal(size=(4, 7))
    assert np.allclose(softmax(x), softmax(x + 123.4))


@needs_torch
def test_softmax_matches_torch():
    x = rng.normal(size=(3, 5, 11)) * 20
    t = torch.tensor(x)
    assert np.allclose(softmax(x), torch.softmax(t, -1).numpy())
    assert np.allclose(log_softmax(x), torch.log_softmax(t, -1).numpy())


@needs_torch
def test_softmax_backward_matches_autograd():
    x = rng.normal(size=(4, 9)); dy = rng.normal(size=(4, 9))
    t = torch.tensor(x, requires_grad=True)
    torch.softmax(t, -1).backward(torch.tensor(dy))
    assert np.allclose(softmax_backward(softmax(x), dy), t.grad.numpy())


# ---------- 2. attention ----------
@needs_torch
def test_sdpa_matches_torch_with_causal_mask():
    q, k, v = (rng.normal(size=(2, 3, 6, 8)) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(6))
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(torch.tensor(a) for a in (q, k, v)), is_causal=True).numpy()
    assert np.allclose(out, ref, atol=1e-10)
    assert np.allclose(np.triu(w[0, 0], 1), 0)          # 미래를 안 봄


@needs_torch
def test_mha_matches_torch_module():
    D, H, B, T = 16, 4, 2, 5
    mine = MultiHeadAttention(D, H)
    ref = torch.nn.MultiheadAttention(D, H, bias=False, batch_first=True).double()
    with torch.no_grad():   # torch는 x @ W^T 규약이라 전치해서 넣는다
        ref.in_proj_weight.copy_(torch.tensor(np.concatenate([mine.W_q.T, mine.W_k.T, mine.W_v.T])))
        ref.out_proj.weight.copy_(torch.tensor(mine.W_o.T))
    x = rng.normal(size=(B, T, D))
    y, _ = mine(x, causal_mask(T))
    xt = torch.tensor(x)
    y_ref, _ = ref(xt, xt, xt, attn_mask=torch.tensor(causal_mask(T)), need_weights=False)
    assert np.allclose(y, y_ref.detach().numpy(), atol=1e-10)


# ---------- 3. k-means ----------
def blobs(seed=0):
    r = np.random.default_rng(seed)
    return np.vstack([r.normal(m, 0.6, (150, 2)) for m in ([0, 0], [5, 5], [0, 5], [5, 0])])


def test_kmeans_inertia_monotone():
    _, _, _, hist = kmeans(blobs(), 4, init="random", seed=3)
    assert all(b <= a + 1e-9 for a, b in zip(hist, hist[1:]))


def test_kmeans_close_to_sklearn():
    sk = pytest.importorskip("sklearn.cluster")
    X = blobs()
    best_mine = min(kmeans(X, 4, seed=s)[2] for s in range(10))
    ref = sk.KMeans(4, n_init=10, random_state=0).fit(X).inertia_
    assert best_mine <= ref * 1.001


# ---------- 4. Adam ----------
@needs_torch
@pytest.mark.parametrize("wd,decoupled", [(0.0, True), (0.1, False), (0.1, True)])
def test_adam_matches_torch(wd, decoupled):
    A = rng.normal(size=(5, 5)); b = rng.normal(size=5)
    p = rng.normal(size=5)
    pt = torch.tensor(p.copy(), requires_grad=True)
    cls = torch.optim.AdamW if decoupled else torch.optim.Adam
    topt = cls([pt], lr=0.05, weight_decay=wd)
    mopt = Adam([p], lr=0.05, weight_decay=wd, decoupled=decoupled)
    At, bt = torch.tensor(A), torch.tensor(b)
    for _ in range(50):
        mopt.step([A.T @ (A @ p - b)])           # f = 0.5||Ap-b||^2
        topt.zero_grad(); (0.5 * ((At @ pt - bt) ** 2).sum()).backward(); topt.step()
    assert np.allclose(p, pt.detach().numpy(), atol=1e-10)


# ---------- 5. beam search ----------
def brute_force_best(lm, bos, eos, V, max_len):
    best = (-np.inf, None)
    for L in range(1, max_len + 1):
        for body in itertools.product(range(1, V), repeat=L - 1):
            seq, lp = (bos,), 0.0
            for tok in body + (eos,):
                lp += lm(seq)[tok]; seq += (tok,)
            best = max(best, (lp, seq))
    return best


def test_beam_beats_greedy_on_classic_example():
    ex = classic_example()
    assert beam_search(ex, 1, 0, beam_width=2)[0][1] > greedy(ex, 1, 0)[0]


def test_wide_beam_equals_brute_force():
    V, L = 4, 4
    for seed in range(5):
        lm = make_toy_lm(V=V, seed=seed)
        lp, seq = brute_force_best(lm, 1, 0, V, L)
        res = [r for r in beam_search(lm, 1, 0, beam_width=V ** L, max_len=L) if r[2][-1] == 0]
        assert np.isclose(res[0][1], lp) and res[0][2] == seq


def test_beam_never_worse_than_greedy_width1():
    for seed in range(20):
        lm = make_toy_lm(seed=seed)
        g = greedy(lm, 1, 0, max_len=8)
        b = beam_search(lm, 1, 0, beam_width=1, max_len=8)[0]
        assert np.isclose(b[1], g[0])               # 빔 폭 1 = greedy
