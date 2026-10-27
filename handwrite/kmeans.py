"""3. K-Means (Lloyd 알고리즘) + k-means++ 초기화 (NumPy).

반복: (1) 각 점을 가장 가까운 중심에 배정 (2) 중심 = 배정된 점들의 평균
- 두 단계 모두 inertia(SSE)를 늘리지 않음 → 단조 감소, 유한 번 안에 수렴 (전역 최적 보장은 아님)
- 빈 클러스터 처리: 가장 먼 점으로 다시 심기
"""
import numpy as np


def pairwise_sq_dist(X, C):
    # ||x-c||^2 = ||x||^2 - 2 x·c + ||c||^2   (브로드캐스팅으로 (N,K))
    return (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]


def kmeans_pp_init(X, k, rng):
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(pairwise_sq_dist(X, np.array(C)), axis=1).clip(min=0)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])   # 멀수록 뽑힐 확률 ↑ (D^2 샘플링)
    return np.array(C)


def kmeans(X, k, n_iter=100, init="k-means++", seed=0, tol=1e-8):
    rng = np.random.default_rng(seed)
    C = kmeans_pp_init(X, k, rng) if init == "k-means++" else X[rng.choice(len(X), k, replace=False)].copy()
    history = []
    for _ in range(n_iter):
        d2 = pairwise_sq_dist(X, C)
        labels = d2.argmin(1)
        history.append(d2[np.arange(len(X)), labels].clip(min=0).sum())
        newC = C.copy()
        for j in range(k):
            pts = X[labels == j]
            if len(pts):
                newC[j] = pts.mean(0)
            else:                                   # 빈 클러스터
                newC[j] = X[d2.min(1).argmax()]
        shift = ((newC - C) ** 2).sum()
        C = newC
        if shift < tol:
            break
    d2 = pairwise_sq_dist(X, C)
    labels = d2.argmin(1)
    inertia = d2[np.arange(len(X)), labels].clip(min=0).sum()
    return C, labels, inertia, history


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 0.5, (100, 2)) for m in ([0, 0], [4, 4], [0, 4])])
    C, labels, inertia, hist = kmeans(X, 3)
    print("centers\n", C.round(2), "\ninertia", round(inertia, 2), "iters", len(hist))
