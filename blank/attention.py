"""연습용 빈칸.  IMPL=blank pytest -q -k "sdpa or mha" """
import numpy as np
from blank.softmax import softmax


def scaled_dot_product_attention(q, k, v, mask=None):
    """mask: True = 가린다. 반환 (out, weights)"""
    raise NotImplementedError


def causal_mask(T):
    raise NotImplementedError


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, rng=None):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(0) if rng is None else rng
        self.h, self.dh = n_heads, d_model // n_heads
        s = 1.0 / np.sqrt(d_model)
        self.W_q, self.W_k, self.W_v, self.W_o = (
            rng.normal(0, s, (d_model, d_model)) for _ in range(4))

    def __call__(self, x, mask=None):
        """x:(B,T,D) → (out (B,T,D), weights (B,H,T,T)). 규약: q = x @ W_q"""
        raise NotImplementedError
