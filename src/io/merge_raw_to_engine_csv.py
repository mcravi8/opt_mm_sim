# src/io/merge_raw_to_engine_csv.py
"""
Merge raw minute underlying + option quotes into an engine-ready CSV.

Inputs (you provide):
  - Underlying CSV: timestamp + a price column (e.g., close/last)
  - Option CSV: timestamp + call bid/ask (and optionally put bid/ask), plus strike/expiry
Outputs:
  - CSV with columns the engine loader understands:
      timestamp, underlying, call_mid, call_bid, call_ask, put_mid, strike, expiry, r, iv, dividend_yield

Example:
  python src/io/merge_raw_to_engine_csv.py \
    --under data/raw/spy_2024-07-01_1m.csv --under-ts-col timestamp --under-price-col close \
    --opt data/raw/spy_2024-07-01_Jul19_K450_quotes.csv --opt-ts-col timestamp \
    --call-bid-col call_bid --call-ask-col call_ask --put-bid-col put_bid --put-ask-col put_ask \
    --strike 450 --expiry 2024-07-19 \
    --out data/SPY_Jul19_K450_merged.csv

Then run:
  python src/engine/backtest_from_csv.py --csv data/SPY_Jul19_K450_merged.csv --k-fill 8 --skew 0.02
"""

import argparse
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    # Underlying
    p.add_argument("--under", required=True, help="Underlying CSV path")
    p.add_argument("--under-ts-col", default="timestamp", help="Underlying timestamp column name")
    p.add_argument("--under-price-col", default="close", help="Underlying price column (close/last)")
    # Options
    p.add_argument("--opt", required=True, help="Options CSV path")
    p.add_argument("--opt-ts-col", default="timestamp", help="Options timestamp column name")
    p.add_argument("--call-bid-col", default="call_bid", help="Call bid column (if present)")
    p.add_argument("--call-ask-col", default="call_ask", help="Call ask column (if present)")
    p.add_argument("--call-mid-col", default="call_mid", help="Call mid column (if present)")
    p.add_argument("--put-bid-col", default="put_bid", help="Put bid column (optional)")
    p.add_argument("--put-ask-col", default="put_ask", help="Put ask column (optional)")
    p.add_argument("--put-mid-col", default="put_mid", help="Put mid column (optional)")
    p.add_argument("--opt-strike-col", default="strike", help="Strike column name (if present)")
    p.add_argument("--opt-expiry-col", default="expiry", help="Expiry column (YYYY-MM-DD) if present")
    # Instrument metadata (overrides if not in CSV)
    p.add_argument("--strike", type=float, default=None, help="Strike (overrides/used if not in CSV)")
    p.add_argument("--expiry", type=str, default=None, help="Expiry YYYY-MM-DD (overrides/used if not in CSV)")
    # Market params
    p.add_argument("--r", type=float, default=0.01, help="Risk-free rate (annual)")
    p.add_argument("--iv", type=float, default=0.25, help="Implied vol (annual) stored for reference")
    p.add_argument("--q", type=float, default=0.0, help="Dividend yield (annual)")
    # Merge behavior
    p.add_argument("--tolerance", default="1min", help="Max timestamp gap for merge_asof (e.g., 1min, 2min)")
    # Output
    p.add_argument("--out", required=True, help="Output CSV path")
    return p.parse_args()

def read_under(path, ts_col, price_col):
    df = pd.read_csv(path)
    if ts_col not in df.columns:
        raise ValueError(f"Underlying timestamp column '{ts_col}' not found.")
    if price_col not in df.columns:
        raise ValueError(f"Underlying price column '{price_col}' not found.")
    df = df[[ts_col, price_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df.rename(columns={ts_col: "timestamp", price_col: "underlying"}, inplace=True)
    df = df.sort_values("timestamp").dropna()
    return df

def read_opts(path, ts_col, cb, ca, cm, pb, pa, pm, strike_col, expiry_col):
    df = pd.read_csv(path)
    if ts_col not in df.columns:
        raise ValueError(f"Options timestamp column '{ts_col}' not found.")
    df[ts_col] = pd.to_datetime(df[ts_col])
    keep_cols = [ts_col]
    rename_map = {}

    # call side
    have_mid = cm in df.columns
    have_bidask = (cb in df.columns) and (ca in df.columns)
    if not have_mid and not have_bidask:
        raise ValueError("Provide either call mid or both call bid & call ask.")
    if have_mid:
        rename_map[cm] = "call_mid"
        keep_cols.append(cm)
    if have_bidask:
        rename_map[cb] = "call_bid"
        rename_map[ca] = "call_ask"
        keep_cols.extend([cb, ca])

    # put side (optional)
    have_put_mid = pm in df.columns
    have_put_bidask = (pb in df.columns) and (pa in df.columns)
    if have_put_mid:
        rename_map[pm] = "put_mid"
        keep_cols.append(pm)
    if have_put_bidask:
        rename_map[pb] = "put_bid"
        rename_map[pa] = "put_ask"
        keep_cols.extend([pb, pa])

    # strike/expiry if present
    if strike_col in df.columns:
        rename_map[strike_col] = "strike"
        keep_cols.append(strike_col)
    if expiry_col in df.columns:
        rename_map[expiry_col] = "expiry"
        keep_cols.append(expiry_col)

    df = df[keep_cols].rename(columns=rename_map)
    df.rename(columns={ts_col: "timestamp"}, inplace=True)
    df = df.sort_values("timestamp").dropna(subset=["timestamp"])

    # derive mids if needed
    if "call_mid" not in df.columns and {"call_bid", "call_ask"} <= set(df.columns):
        df["call_mid"] = 0.5 * (df["call_bid"] + df["call_ask"])
    if ("put_mid" not in df.columns) and ({"put_bid", "put_ask"} <= set(df.columns)):
        df["put_mid"] = 0.5 * (df["put_bid"] + df["put_ask"])

    return df

def main():
    a = parse_args()

    under = read_under(a.under, a.under_ts_col, a.under_price_col)
    opts = read_opts(
        a.opt, a.opt_ts_col,
        a.call_bid_col, a.call_ask_col, a.call_mid_col,
        a.put_bid_col, a.put_ask_col, a.put_mid_col,
        a.opt_strike_col, a.opt_expiry_col
    )

    # Merge with nearest timestamp within tolerance
    merged = pd.merge_asof(
        under, opts,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(a.tolerance)
    )

    # Drop rows where we still don't have call_mid after merge
    merged = merged.dropna(subset=["call_mid"]).copy()

    # Fill strike/expiry if not present or overridden
    if "strike" not in merged.columns or a.strike is not None:
        merged["strike"] = a.strike if a.strike is not None else merged.get("strike", np.nan).ffill().bfill()
    if "expiry" not in merged.columns or a.expiry is not None:
        merged["expiry"] = a.expiry if a.expiry is not None else merged.get("expiry", np.nan).ffill().bfill()

    # Market params
    merged["r"] = float(a.r)
    merged["iv"] = float(a.iv)
    merged["dividend_yield"] = float(a.q)

    # Reorder/save minimal useful set
    out_cols = [
        "timestamp",
        "underlying",
        # call side
        "call_mid",
        "call_bid" if "call_bid" in merged.columns else None,
        "call_ask" if "call_ask" in merged.columns else None,
        # put (optional)
        "put_mid" if "put_mid" in merged.columns else None,
        # instrument + params
        "strike", "expiry", "r", "iv", "dividend_yield"
    ]
    out_cols = [c for c in out_cols if c is not None and c in merged.columns]

    merged[out_cols].to_csv(a.out, index=False)
    print(f"Wrote {a.out} with {len(merged)} rows and columns: {out_cols}")

if __name__ == "__main__":
    main()
