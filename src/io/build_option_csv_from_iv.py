# src/io/build_option_csv_from_iv.py
"""
Build a minute option quotes CSV from underlying minute prices using Black–Scholes.

Inputs:
  --under: CSV with columns [timestamp, close] (UTC or naive ok)
  --strike: option strike
  --expiry: YYYY-MM-DD (option expiry date)
  --iv: implied vol (annual)
  --r: risk-free (annual), --q: dividend yield
  --spread: absolute option spread OR --spread-pct of mid

Output CSV columns:
  timestamp, call_mid, call_bid, call_ask, put_mid, strike, expiry, r, iv, dividend_yield
"""
import argparse, math
from datetime import datetime
import numpy as np
import pandas as pd

def norm_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_price(S, K, r, sig, T, q=0.0):
    if T <= 0 or sig <= 0: return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r - q + 0.5*sig*sig)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    # with dividends q: e^{-qT} * S * N(d1) - K e^{-rT} N(d2)
    return math.exp(-q*T)*S*norm_cdf(d1) - K*math.exp(-r*T)*norm_cdf(d2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--under", required=True, help="Underlying CSV (timestamp,close)")
    p.add_argument("--strike", type=float, required=True)
    p.add_argument("--expiry", required=True, help="YYYY-MM-DD")
    p.add_argument("--iv", type=float, default=0.22)
    p.add_argument("--r", type=float, default=0.01)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--spread", type=float, default=None, help="Absolute spread in option price units")
    p.add_argument("--spread-pct", type=float, default=0.03, help="If --spread not set, use pct of mid (e.g., 0.03)")
    p.add_argument("--out", required=True, help="Output CSV path")
    args = p.parse_args()

    # Load underlying
    df = pd.read_csv(args.under, parse_dates=["timestamp"])
    if df.empty:
        raise SystemExit("Underlying CSV empty.")

    # Normalize timestamps to tz-aware UTC for safe arithmetic
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts

    # Make expiry tz-aware UTC as well
    expiry_dt = pd.to_datetime(args.expiry).tz_localize("UTC")

    # Time to expiry in years (trading-day approx: 252)
    secs = (expiry_dt - df["timestamp"]).dt.total_seconds().to_numpy()
    T_years = np.clip(secs / (86400.0 * 252.0), 1e-9, None)

    # Inputs
    S = df["close"].astype(float).to_numpy()
    K, r, q, iv = float(args.strike), float(args.r), float(args.q), float(args.iv)

    # Call mid via BS
    call_mid = np.array([bs_call_price(S[i], K, r, iv, T_years[i], q) for i in range(len(df))])

    # Build bid/ask
    if args.spread is not None:
        half = float(args.spread) / 2.0
        call_bid = call_mid - half
        call_ask = call_mid + half
    else:
        # percentage of mid with a small floor
        half_arr = np.maximum(0.01, args.spread_pct * np.maximum(call_mid, 0.25)) / 2.0
        call_bid = call_mid - half_arr
        call_ask = call_mid + half_arr

    # Put mid from parity: P = C - (S - PV(K))
    pvK = K * np.exp(-r * T_years)
    put_mid = call_mid - (S - pvK)

    out = pd.DataFrame({
        "timestamp": df["timestamp"],     # remains UTC tz-aware
        "call_mid": call_mid,
        "call_bid": call_bid,
        "call_ask": call_ask,
        "put_mid": put_mid,
        "strike": K,
        "expiry": args.expiry,
        "r": r,
        "iv": iv,
        "dividend_yield": q
    })
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows")

if __name__ == "__main__":
    main()
