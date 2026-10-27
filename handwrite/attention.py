"""2. Scaled Dot-Product Attention + Multi-Head Attention (NumPy).

shape 규칙 (면접에서 제일 많이 틀리는 곳)
  x      : (B, T, D)
  W_q,k,v: (D, D)       → q,k,v: (B, T, D)
  split  : (B, T, H, Dh) → transpose → (B, H, T, Dh),  Dh = D // H
  scores : (B, H, T, T) = q @ k^T / sqrt(Dh)
  out    : (B, H, T, Dh) → (B, T, H, Dh) → (B, T, D) → @ W_o
"""
import numpy as np
from handwrite.softmax import softmax


def scaled_dot_product_attention(q, k, v, mask=None):
    """q:(...,Tq,d) k:(...,Tk,d) v:(...,Tk,dv). mask: True = 가린다(보지 못함)."""
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)     # -inf 대신 큰 음수: 행 전체가 가려져도 NaN 안 남
    w = softmax(scores, axis=-1)
    return w @ v, w


def causal_mask(T):
    """미래 토큰을 가리는 상삼각 마스크 (대각선 위가 True)."""
    return np.triu(np.ones((T, T), dtype=bool), k=1)


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, rng=None):
        assert d_model % n_heads == 0, "d_model은 n_heads로 나눠떨어져야 함"
        rng = np.random.default_rng(0) if rng is None else rng
        self.h, self.dh = n_heads, d_model // n_heads
        s = 1.0 / np.sqrt(d_model)
        self.W_q, self.W_k, self.W_v, self.W_o = (
            rng.normal(0, s, (d_model, d_model)) for _ in range(4))

    def _split(self, x):                      # (B,T,D) -> (B,H,T,Dh)
        B, T, _ = x.shape
        return x.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)

    def _merge(self, x):                      # (B,H,T,Dh) -> (B,T,D)
        B, H, T, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

    def __call__(self, x, mask=None):
        q, k, v = (self._split(x @ W) for W in (self.W_q, self.W_k, self.W_v))
        out, w = scaled_dot_product_attention(q, k, v, mask)
        return self._merge(out) @ self.W_o, w


if __name__ == "__main__":
    x = np.random.default_rng(1).normal(size=(2, 5, 16))
    mha = MultiHeadAttention(16, 4)
    y, w = mha(x, mask=causal_mask(5))
    print("out", y.shape, "weights", w.shape, "row sums", w.sum(-1)[0, 0])
