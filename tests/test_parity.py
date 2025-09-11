# tests/test_parity.py
from math import exp
from src.theory.parity import put_call_parity_residual

def test_parity_synthetic():
    S, K, r, T = 100.0, 100.0, 0.01, 30/252
    pvK = K * exp(-r * T)
    P = 1.2
    C = P + S - pvK
    res = put_call_parity_residual(C, P, S, K, r, T)
    assert abs(res) < 1e-9
