"""연습용 빈칸.  IMPL=blank pytest -q -k adam"""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 decoupled=True, bias_correction=True):
        raise NotImplementedError

    def step(self, grads):
        """params를 제자리(in-place)로 갱신. decoupled=True면 AdamW."""
        raise NotImplementedError
