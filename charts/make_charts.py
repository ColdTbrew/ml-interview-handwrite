"""README에 들어가는 그림을 전부 다시 만든다:  python charts/make_charts.py"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in fm.findSystemFonts():
    if "NanumGothic" in f:
        fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["DejaVu Sans", "NanumGothic"], "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
                     "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
OUT = ROOT / "charts"

from handwrite.softmax import softmax, softmax_naive
from handwrite.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from handwrite.kmeans import kmeans
from handwrite.adam import Adam, SGD
from handwrite.beam_search import beam_search, greedy, make_toy_lm


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig); print("wrote", name)


# 1 softmax ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
scales = np.logspace(0, 3.2, 60)
base = np.array([0.1, 0.5, 1.0])
naive_ok, stable_ok = [], []
with np.errstate(over="ignore", invalid="ignore"):
    for s in scales:
        naive_ok.append(np.nan_to_num(softmax_naive(base * s)[-1], nan=-0.05))
        stable_ok.append(softmax(base * s)[-1])
ax[0].plot(scales, stable_ok, lw=2.5, label="안정 버전 (x - max)")
ax[0].plot(scales, naive_ok, "--", lw=2, label="naive exp(x) (NaN은 -0.05로 표시)")
ax[0].axvline(709.78, color="gray", ls=":"); ax[0].text(720, 0.4, "exp(710) = inf\n(float64 한계)", fontsize=9)
ax[0].set_xscale("log"); ax[0].set_xlabel("logit 크기 배율"); ax[0].set_ylabel("마지막 클래스 확률")
ax[0].set_title("최댓값을 안 빼면 큰 logit에서 NaN"); ax[0].legend(fontsize=9)
logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
w = 0.15
for i, T in enumerate([0.3, 1.0, 3.0]):
    ax[1].bar(np.arange(5) + (i - 1) * w * 1.6, softmax(logits, temperature=T), width=w * 1.5, label=f"T={T}")
ax[1].set_xticks(range(5), [f"logit {l:g}" for l in logits]); ax[1].set_ylabel("확률")
ax[1].set_title("온도 T: 작으면 뾰족(argmax), 크면 평평(균등)"); ax[1].legend()
save(fig, "01_softmax.png")

# 2 attention ---------------------------------------------------------------
fig = plt.figure(figsize=(13, 4.2))
toks = ["나는", "오늘", "논문을", "읽고", "정리했다", "."]
T = len(toks)
x = np.random.default_rng(7).normal(size=(1, T, 32))
_, wts = MultiHeadAttention(32, 4, rng=np.random.default_rng(3))(x, causal_mask(T))
for h in range(4):
    a = fig.add_subplot(1, 5, h + 1)
    a.imshow(wts[0, h], cmap="viridis", vmin=0, vmax=1)
    a.set_title(f"head {h}"); a.set_xticks(range(T), toks, rotation=60, fontsize=8)
    a.set_yticks(range(T), toks if h == 0 else [""] * T, fontsize=8)
    for s in a.spines.values(): s.set_visible(False)
a = fig.add_subplot(1, 5, 5)
ds = [4, 16, 64, 256, 1024]
r = np.random.default_rng(0)
ent_s, ent_u = [], []
for d in ds:
    q = r.normal(size=(200, 1, d)); k = r.normal(size=(200, 32, d))
    for scale, store in ((True, ent_s), (False, ent_u)):
        sc = (q @ k.transpose(0, 2, 1))[:, 0] / (np.sqrt(d) if scale else 1)
        p = softmax(sc); store.append(-(p * np.log(p + 1e-12)).sum(-1).mean())
a.plot(ds, ent_s, "o-", label="÷√d 함"); a.plot(ds, ent_u, "s--", label="÷√d 안 함")
a.axhline(np.log(32), color="gray", ls=":"); a.text(4, np.log(32) - 0.35, "균등(최대)", fontsize=8)
a.set_xscale("log", base=2); a.set_xlabel("head 차원 d"); a.set_ylabel("attention 엔트로피")
a.set_title("√d로 안 나누면 one-hot으로 쏠림"); a.legend(fontsize=8)
fig.suptitle("Causal Multi-Head Attention 가중치 (무작위 가중치, 아래 삼각형만 0이 아님)", y=1.02)
save(fig, "02_attention.png")

# 3 k-means ----------------------------------------------------------------
r = np.random.default_rng(1)
centers = np.array([[0, 0], [6, 0], [3, 5], [9, 5], [6, 10], [0, 10]])
X = np.vstack([r.normal(c, 0.8, (80, 2)) for c in centers])
res = {m: [kmeans(X, 6, init=m, seed=s) for s in range(60)] for m in ("random", "k-means++")}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].boxplot([[r_[2] for r_ in res[m]] for m in res], tick_labels=["무작위 초기화", "k-means++"])
best = min(r_[2] for m in res for r_ in res[m])
for i, m in enumerate(res):
    ok = np.mean([r_[2] <= best * 1.01 for r_ in res[m]]) * 100
    ax[0].text(i + 1, best * 1.6, f"최적해 도달\n{ok:.0f}%", ha="center", fontsize=10, color="C3")
ax[0].set_ylabel("최종 inertia (SSE)"); ax[0].set_title("초기화 60번: k-means++가 나쁜 국소해를 덜 밟음")
for m, c in (("random", "C1"), ("k-means++", "C0")):
    for r_ in res[m][:15]:
        ax[1].plot(r_[3], color=c, alpha=.35)
    ax[1].plot([], color=c, label=m)
ax[1].set_yscale("log"); ax[1].set_xlabel("반복"); ax[1].set_ylabel("inertia"); ax[1].legend()
ax[1].set_title("inertia는 매 반복 단조 감소")
worst = max(res["random"], key=lambda r_: r_[2])
ax[2].scatter(X[:, 0], X[:, 1], c=worst[1], cmap="tab10", s=8)
ax[2].scatter(*worst[0].T, c="k", marker="x", s=120)
ax[2].set_title(f"무작위 초기화 최악 사례 (inertia {worst[2]:.0f})\n두 덩어리를 한 중심이 먹음")
save(fig, "03_kmeans.png")

# 4 Adam -------------------------------------------------------------------
def rosen_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])
def run(make, steps=3000):
    p = np.array([-1.5, 2.0]); opt = make([p]); path = [p.copy()]
    for _ in range(steps):
        opt.step([rosen_grad(p)]); path.append(p.copy())
    return np.array(path)
paths = {"SGD lr=1e-3": run(lambda ps: SGD(ps, lr=1e-3)),
         "Momentum 0.9 lr=2e-4": run(lambda ps: SGD(ps, lr=2e-4, momentum=0.9)),
         "Adam lr=0.02": run(lambda ps: Adam(ps, lr=0.02))}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
gx, gy = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1, 3, 300))
ax[0].contour(gx, gy, np.log1p((1 - gx) ** 2 + 100 * (gy - gx ** 2) ** 2), 25, cmap="Greys", linewidths=.6)
for (n, pth), c in zip(paths.items(), ["C1", "C2", "C0"]):
    ax[0].plot(pth[:, 0], pth[:, 1], color=c, lw={"C0": 4, "C2": 2.2, "C1": 1.2}[c], alpha=.8, label=n)
    ax[0].plot(*pth[-1], "o", color=c, ms=7)
ax[0].plot(1, 1, "r*", ms=14); ax[0].set_title("Rosenbrock 3000스텝, 시작 (-1.5, 2) → ★(1,1)\n(lr은 손으로 고름: 공정한 벤치마크 아님)"); ax[0].legend(fontsize=8)
for (n, pth), c in zip(paths.items(), ["C1", "C2", "C0"]):
    f = (1 - pth[:, 0]) ** 2 + 100 * (pth[:, 1] - pth[:, 0] ** 2) ** 2
    ax[1].plot(f + 1e-12, color=c, label=n)
ax[1].set_yscale("log"); ax[1].set_xlabel("스텝"); ax[1].set_ylabel("loss"); ax[1].legend(fontsize=8)
ax[1].set_title("loss 곡선")
# 편향 보정: 상수 그래디언트 g=1일 때 실제 스텝 크기
for bc, ls in ((True, "-"), (False, "--")):
    p = np.zeros(1); o = Adam([p], lr=1.0, bias_correction=bc); steps = []
    for _ in range(60):
        before = p.copy(); o.step([np.ones(1)]); steps.append(float((before - p)[0]))
    ax[2].plot(range(1, 61), steps, ls, lw=2, label=f"편향 보정 {'O' if bc else 'X'}")
ax[2].set_xlabel("스텝 t"); ax[2].set_ylabel("실제 스텝 / lr"); ax[2].legend()
ax[2].set_title("g=1 고정일 때: 보정 없으면 첫 스텝이 (1-β1)/√(1-β2) ≈ 3.16·lr,\n이후 최대 ≈6.5·lr까지 커짐 (보정하면 정확히 lr)")
save(fig, "04_adam.png")

# 5 beam search ------------------------------------------------------------
widths = [1, 2, 3, 5, 8, 16]
N = 200
gaps, better = {w: [] for w in widths}, {w: 0 for w in widths}
for seed in range(N):
    lm = make_toy_lm(V=8, seed=seed)
    g = greedy(lm, 1, 0, max_len=8)[0]
    for w in widths:
        b = beam_search(lm, 1, 0, beam_width=w, max_len=8)[0][1]
        gaps[w].append(b - g); better[w] += b > g + 1e-9
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(widths, [np.mean(gaps[w]) for w in widths], "o-", lw=2)
ax[0].set_xlabel("빔 폭 B"); ax[0].set_ylabel("greedy 대비 평균 log-prob 이득")
ax[0].set_title(f"무작위 toy LM {N}개 (V=8, 최대 8토큰)")
ax[1].bar([str(w) for w in widths], [100 * better[w] / N for w in widths], color="C0")
ax[1].set_xlabel("빔 폭 B"); ax[1].set_ylabel("greedy보다 높은 문장을 찾은 비율 (%)")
ax[1].set_title("이득은 B가 작을 때 대부분 생기고 곧 포화")
save(fig, "05_beam_search.png")
