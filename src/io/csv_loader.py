# src/io/csv_loader.py
"""
CSV loader for a single option series (one strike & expiry) at minute frequency.

Required CSV columns (case-insensitive names are OK):
- underlying (or S)
- strike (or K)
- EITHER call_mid  OR BOTH call_bid & call_ask
- ONE of: expiry (YYYY-MM-DD) OR time_to_expiry_days OR T_years
Optional:
- timestamp (recommended; UTC or naive is OK)
- put_mid (enables parity residual)
- r (risk-free, annual)  iv (implied vol, annual)  dividend_yield (q)  pv_div
- spread (if no bid/ask spread available)
"""

import math
import numpy as np
import pandas as pd

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_call_delta(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    """Black–Scholes call delta with continuous dividend yield q."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * _norm_cdf(d1)

def load_option_csv(
    path: str,
    default_r: float = 0.01,
    default_iv: float = 0.25,
    default_q: float = 0.0,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    # lower-case lookup map
    cols = {c.lower(): c for c in df.columns}

    # ----- timestamps -----
    ts_col = cols.get("timestamp")  # optional but recommended
    if ts_col:
        # Make timestamp tz-aware UTC
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    # required: underlying
    S_col = cols.get("underlying", cols.get("s"))
    if S_col is None:
        raise ValueError("CSV must include 'underlying' (or 'S').")
    df.rename(columns={S_col: "underlying"}, inplace=True)

    # required: strike
    K_col = cols.get("strike", cols.get("k"))
    if K_col is None:
        raise ValueError("CSV must include 'strike' (or 'K').")
    df.rename(columns={K_col: "strike"}, inplace=True)

    # call prices: mid OR (bid+ask)
    call_mid_col = cols.get("call_mid")
    call_bid_col = cols.get("call_bid")
    call_ask_col = cols.get("call_ask")
    if call_mid_col:
        df.rename(columns={call_mid_col: "call_mid"}, inplace=True)
    elif call_bid_col and call_ask_col:
        df.rename(columns={call_bid_col: "call_bid", call_ask_col: "call_ask"}, inplace=True)
        df["call_mid"] = 0.5 * (df["call_bid"] + df["call_ask"])
    else:
        raise ValueError("Need 'call_mid' or both 'call_bid' and 'call_ask'.")

    # ----- T_years from provided info -----
    expiry_col = cols.get("expiry")
    tdays_col = cols.get("time_to_expiry_days")
    tyears_col = cols.get("t_years")

    if tyears_col:
        df.rename(columns={tyears_col: "T_years"}, inplace=True)

    elif tdays_col:
        df.rename(columns={tdays_col: "time_to_expiry_days"}, inplace=True)
        df["T_years"] = df["time_to_expiry_days"] / 252.0

    elif expiry_col and ts_col:
        # Normalize expiry to UTC tz-aware, same as timestamp
        df.rename(columns={expiry_col: "expiry"}, inplace=True)
        df["expiry"] = pd.to_datetime(df["expiry"], utc=True, errors="coerce")
        # Now both are tz-aware UTC, subtraction is safe
        dt_secs = (df["expiry"] - df[ts_col]).dt.total_seconds()
        df["T_years"] = (dt_secs / (86400.0 * 252.0)).clip(lower=1e-6)

    elif expiry_col and not ts_col:
        # No timestamp column → approximate linear decay from 30D
        df.rename(columns={expiry_col: "expiry"}, inplace=True)
        n = len(df)
        T0 = 30.0 / 252.0
        df["T_years"] = (T0 * (1 - (df.reset_index().index / max(1, n - 1)))).clip(lower=1e-6)

    else:
        # Fallback: constant 30 days
        df["T_years"] = 30.0 / 252.0

    # ----- optional: r / iv / q / pv_div -----
    if cols.get("r"): df.rename(columns={cols["r"]: "r"}, inplace=True)
    else: df["r"] = default_r

    if cols.get("iv"): df.rename(columns={cols["iv"]: "iv"}, inplace=True)
    else: df["iv"] = default_iv

    if cols.get("dividend_yield"): df.rename(columns={cols["dividend_yield"]: "q"}, inplace=True)
    else: df["q"] = default_q

    if cols.get("pv_div"): df.rename(columns={cols["pv_div"]: "pv_div"}, inplace=True)
    else: df["pv_div"] = 0.0

    # spread
    if cols.get("spread"):
        df.rename(columns={cols["spread"]: "spread"}, inplace=True)
    elif "call_bid" in df and "call_ask" in df:
        df["spread"] = (df["call_ask"] - df["call_bid"]).abs().clip(lower=1e-6)
    else:
        df["spread"] = 0.05  # safe default

    # optional: put_mid for parity residual
    put_mid_col = cols.get("put_mid")
    if put_mid_col:
        df.rename(columns={put_mid_col: "put_mid"}, inplace=True)

    # ----- Build engine-ready DataFrame -----
    out = pd.DataFrame()
    out["minute"] = range(len(df))
    out["underlying"] = df["underlying"].astype(float)
    out["mid"] = df["call_mid"].astype(float)
    out["spread"] = df["spread"].astype(float)
    out["T_years"] = df["T_years"].astype(float)

    # theo delta (BS with optional dividends)
    out["theo_delta"] = [
        _bs_call_delta(
            S=float(df["underlying"].iloc[i]),
            K=float(df["strike"].iloc[i]),
            r=float(df["r"].iloc[i]),
            sigma=float(df["iv"].iloc[i]),
            T=float(df["T_years"].iloc[i]),
            q=float(df["q"].iloc[i]),
        )
        for i in range(len(df))
    ]

    # parity residual if we have put_mid: C - (P + S - PV(K) - PV(div))
    if "put_mid" in df.columns:
        pvK = df["strike"].astype(float) * np.exp(-df["r"].astype(float) * df["T_years"].astype(float))
        out["parity_residual"] = df["call_mid"].astype(float) - (
            df["put_mid"].astype(float) + df["underlying"].astype(float) - pvK - df["pv_div"].astype(float)
        )
    else:
        out["parity_residual"] = 0.0

    # small ML-ish score: short returns + (negative) parity residual z
    ret_short = pd.Series(out["underlying"]).pct_change().fillna(0).rolling(3, min_periods=1).mean().fillna(0)
    pr = out["parity_residual"]
    pr_mad = (pr - pr.mean()).abs().mean() + 1e-9
    pr_z = (pr - pr.median()) / pr_mad
    ml_raw = 0.7 * ret_short.values + 0.3 * (-pr_z.values)
    out["ml_score"] = (ml_raw - ml_raw.mean()) / (ml_raw.std() + 1e-9)

    return out
