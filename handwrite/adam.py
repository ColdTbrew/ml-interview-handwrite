"""4. Adam 옵티마이저 (NumPy).

m_t = b1*m + (1-b1)*g             # 1차 모멘트 (방향)
v_t = b2*v + (1-b2)*g^2           # 2차 모멘트 (크기)
m_hat = m_t/(1-b1^t), v_hat = v_t/(1-b2^t)   # 편향 보정: 0으로 시작해서 초반에 작게 나오는 걸 보정
p -= lr * m_hat / (sqrt(v_hat) + eps)
AdamW: 가중치 감쇠를 그래디언트에 섞지 않고 p -= lr*wd*p 로 따로 뺀다.
"""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 decoupled=True, bias_correction=True):
        self.params = params                      # list of np.ndarray (제자리 갱신)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.wd, self.decoupled, self.bc = weight_decay, decoupled, bias_correction
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if self.wd and not self.decoupled:    # 원래 Adam의 L2: 그래디언트에 더함
                g = g + self.wd * p
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            if self.bc:
                m_hat = m / (1 - self.b1 ** self.t)
                v_hat = v / (1 - self.b2 ** self.t)
            else:
                m_hat, v_hat = m, v
            if self.wd and self.decoupled:        # AdamW
                p -= self.lr * self.wd * p
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:
    def __init__(self, params, lr=1e-3, momentum=0.0):
        self.params, self.lr, self.mu = params, lr, momentum
        self.buf = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for p, g, b in zip(self.params, grads, self.buf):
            b *= self.mu; b += g
            p -= self.lr * b


if __name__ == "__main__":
    # f(x,y) = (x-3)^2 + 10(y+1)^2 최소화
    p = np.array([0.0, 0.0])
    opt = Adam([p], lr=0.1)
    for _ in range(500):
        opt.step([np.array([2 * (p[0] - 3), 20 * (p[1] + 1)])])
    print("argmin ≈", p.round(4))
