# src/io/fetch_underlying_yf.py
"""
Fetch intraday underlying bars via yfinance and save CSV with:
  timestamp,close

Usage:
  python src/io/fetch_underlying_yf.py --ticker SPY --date 2025-09-10 --interval 1m \
    --out data/raw/SPY_2025-09-10_1m.csv

Notes:
- yfinance only serves 1m intraday up to ~30 days back. For older dates use --interval 5m.
- We request start/end as DATEs (YYYY-MM-DD) to avoid parsing issues, then filter to 09:30–16:00 ET.
"""
import argparse
from datetime import datetime, timedelta, time
import pandas as pd
import yfinance as yf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--date", required=True, help="Trading date, e.g. 2025-09-10 (exchange local date)")
    p.add_argument("--interval", default="1m", help="1m or 5m")
    p.add_argument("--out", required=True, help="Output CSV path")
    args = p.parse_args()

    # yfinance behaves best with DATE strings for start/end
    d0 = datetime.fromisoformat(args.date).date()
    start_date = d0.isoformat()
    end_date = (d0 + timedelta(days=1)).isoformat()

    df = yf.Ticker(args.ticker).history(
        start=start_date, end=end_date, interval=args.interval, actions=False
    )
    if df is None or df.empty:
        raise SystemExit("No data returned (try --interval 5m or a more recent date).")

    df = df.reset_index()
    # Handle different column names across yfinance versions
    if "Datetime" in df.columns:
        ts_col = "Datetime"
    elif "Date" in df.columns:
        ts_col = "Date"
    else:
        raise SystemExit("Could not find timestamp column in yfinance output.")

    df.rename(columns={ts_col: "timestamp", "Close": "close"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Convert to US/Eastern to filter regular trading hours
    try:
        ts_et = df["timestamp"].dt.tz_convert("America/New_York")
    except Exception:
        # If tz-naive, localize to UTC then convert
        ts_et = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")

    # Keep 09:30–16:00 ET
    mask = (ts_et.dt.time >= time(9, 30)) & (ts_et.dt.time <= time(16, 0))
    out = df.loc[mask, ["timestamp", "close"]].copy()  # keep UTC timestamps
    if out.empty:
        # Fallback: if filtering emptied the frame, just keep all rows
        out = df[["timestamp", "close"]].copy()

    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows")

if __name__ == "__main__":
    main()
