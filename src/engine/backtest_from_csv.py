# src/engine/backtest_from_csv.py
"""
Run the market-making engine using a real sample CSV instead of synthetic data.

Usage:
  python src/engine/backtest_from_csv.py --csv data/sample_calls.csv \
      --r 0.01 --iv 0.25 --q 0.00 --skew 0.02 --k-fill 50 --seed 123
"""

import os
import sys
import argparse

# Ensure project root in sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.io.csv_loader import load_option_csv
from src.engine.backtest_engine import run_engine  # reuse the same engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV file with real sample")
    p.add_argument("--r", type=float, default=0.01, help="risk-free rate (annual)")
    p.add_argument("--iv", type=float, default=0.25, help="implied vol (annual)")
    p.add_argument("--q", type=float, default=0.0, help="dividend yield (annual)")
    p.add_argument("--skew", type=float, default=0.02, help="quote skew factor")
    p.add_argument("--k-fill", type=float, default=50.0, dest="k_fill",
                   help="fill-probability sensitivity (higher=fewer fills)")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    df = load_option_csv(args.csv, default_r=args.r, default_iv=args.iv, default_q=args.q)
    summary = run_engine(
        df,
        reports_dir="reports",
        seed=args.seed,
        skew_factor=args.skew,
        k_fill=args.k_fill,
    )
    print("SUMMARY (CSV):", summary)


if __name__ == "__main__":
    main()
