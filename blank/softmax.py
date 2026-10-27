"""연습용 빈칸. 종이에 먼저 쓰고 → 여기 옮겨서 → IMPL=blank pytest -q -k softmax"""
import numpy as np


def softmax_naive(x, axis=-1):
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax(x, axis=-1, temperature=1.0):
    raise NotImplementedError("힌트: 최댓값을 빼라")


def log_softmax(x, axis=-1):
    raise NotImplementedError("힌트: log-sum-exp")


def softmax_backward(y, dy, axis=-1):
    raise NotImplementedError("힌트: 야코비안 diag(y) - y y^T 를 행렬 없이")
