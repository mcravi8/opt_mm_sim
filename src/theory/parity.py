# src/theory/parity.py
import math

def pv_cash(amount, r, T):
    """Present value of cash discounted at continuous rate r over T years."""
    return amount * math.exp(-r * T)

def put_call_parity_residual(call_price, put_price, S, K, r, T, pv_div=0.0):
    """
    Compute put-call parity residual:
    residual = C - (P + S - K*exp(-rT) - PV(div))
    If residual is large relative to spreads, it could indicate stale quotes or mispricing.
    """
    pvK = K * math.exp(-r * T)
    rhs = put_price + S - pvK - pv_div
    residual = call_price - rhs
    return residual

