"""1. Softmax — 수치 안정 버전 (NumPy만 사용).

면접 포인트
- 최댓값을 빼도 결과가 같다: softmax(x) = softmax(x - c)  (분자·분모에 e^{-c}가 같이 곱해짐)
- 빼지 않으면 x가 크면 exp가 inf → inf/inf = NaN
- log_softmax는 x - max - log(sum(exp(x - max)))  (log-sum-exp 트릭)
"""
import numpy as np


def softmax_naive(x, axis=-1):
    """일부러 틀리게 쓴 버전: 큰 값에서 overflow."""
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax(x, axis=-1, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / temperature
    x = x - x.max(axis=axis, keepdims=True)       # 핵심 한 줄
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def softmax_backward(y, dy, axis=-1):
    """y = softmax(x) 일 때 dL/dx = y * (dy - sum(dy * y)).  (야코비안 diag(y) - y y^T)"""
    return y * (dy - (dy * y).sum(axis=axis, keepdims=True))


if __name__ == "__main__":
    x = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        print("naive :", softmax_naive(x))
    print("stable:", softmax(x))
    print("sum   :", softmax(x).sum())
