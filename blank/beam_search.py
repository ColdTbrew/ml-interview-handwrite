"""연습용 빈칸.  IMPL=blank pytest -q -k beam"""
import numpy as np
from handwrite.beam_search import make_toy_lm, classic_example  # 테스트용 LM은 그대로 씀


def beam_search(step_logprobs, bos, eos, beam_width=3, max_len=10, alpha=0.0):
    """반환: [(정규화 점수, log-prob 합, 시퀀스 tuple)] 점수 내림차순"""
    raise NotImplementedError


def greedy(step_logprobs, bos, eos, max_len=10):
    """반환: (log-prob 합, 시퀀스 tuple)"""
    raise NotImplementedError
