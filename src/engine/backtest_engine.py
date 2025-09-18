# src/engine/backtest_engine.py
"""
Delta-neutral options market-making demo engine.

What this script does (demo mode):
  • builds synthetic minute data for one option/underlying
  • creates features (incl. parity residual)
  • posts quotes, simulates fills, delta-hedges
  • saves 4 figures to reports/

You can now tune:
  --skew <float>     quote skew factor (default 0.02)
  --k-fill <float>   fill sensitivity (higher = fewer fills; default 50.0)

Run demo:
  python src/engine/backtest_engine.py --demo --n 800 --skew 0.02 --k-fill 50
"""

import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root on sys.path so "from src...." works when running directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# (Imported for completeness; not strictly required in demo.)
from src.theory.parity import put_call_parity_residual  # noqa: F401


# ---------- Black–Scholes helpers ----------
def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_call_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


# ---------- Synthetic data builder ----------
def build_synthetic(n_minutes: int = 800) -> pd.DataFrame:
    """
    Create a synthetic minute-level series for underlying & a single option.
    Returns a DataFrame with columns the engine expects.
    """
    np.random.seed(42)

    S0 = 100.0
    mu = 0.0
    sigma_under = 0.02
    dt = 1 / 390  # minute fraction of a trading day
    returns = np.random.normal(mu * dt, sigma_under * math.sqrt(dt), size=n_minutes)
    S = S0 * np.exp(np.cumsum(returns))

    K = S0
    r = 0.01
    implied_vol = 0.25
    minutes_per_day = 390
    T0_days = 30
    T0 = T0_days * minutes_per_day
    minutes_until_expiry = np.linspace(T0, T0 - (n_minutes - 1), n_minutes)
    T_years = np.maximum(minutes_until_expiry / (252 * minutes_per_day), 1e-6)

    df = pd.DataFrame(
        {"minute": np.arange(n_minutes), "underlying": S, "T_years": T_years}
    )
    df["theo_call"] = [
        bs_call_price(S[i], K, r, implied_vol, T_years[i]) for i in range(n_minutes)
    ]
    df["theo_delta"] = [
        bs_call_delta(S[i], K, r, implied_vol, T_years[i]) for i in range(n_minutes)
    ]

    # Market mid = theo + noise; spread ~ recent realized vol (toy)
    noise_scale = 0.02
    df["mid"] = df["theo_call"] * (1 + noise_scale * np.random.normal(0, 1, n_minutes))
    vol_proxy = (
        pd.Series(df["underlying"]).pct_change().rolling(5, min_periods=1).std().fillna(0)
    )
    df["spread"] = np.maximum(0.01, 0.02 + 5 * vol_proxy.values)

    # Synthetic parity residual (via noisy put)
    pvK = K * np.exp(-r * df["T_years"])
    df["put_synthetic"] = (
        df["mid"] - df["underlying"] + pvK + np.random.normal(0, 0.005, n_minutes) * df["mid"]
    )
    df["parity_residual"] = df["mid"] - (df["put_synthetic"] + df["underlying"] - pvK)

    # Tiny ML-ish score: short returns + negative parity residual z-score
    ret_short = (
        pd.Series(df["underlying"]).pct_change().fillna(0).rolling(3, min_periods=1).mean().fillna(0)
    )
    # pandas.Series.mad() is removed; use robust substitute
    pr = df["parity_residual"]
    pr_mad = (pr - pr.mean()).abs().mean() + 1e-9
    res_z = (pr - pr.median()) / pr_mad
    ml_raw = 0.7 * ret_short.values + 0.3 * (-res_z.values)
    df["ml_score"] = (ml_raw - np.mean(ml_raw)) / (np.std(ml_raw) + 1e-9)

    return df


# ---------- Engine ----------
def run_engine(
    df: pd.DataFrame,
    reports_dir: str = "reports",
    seed: int = 123,
    *,
    skew_factor: float = 0.02,
    k_fill: float = 50.0,
) -> dict:
    """
    Core backtest loop (quotes -> fills -> hedge). Saves plots and returns summary.
    Tunables:
      - skew_factor: how strongly the ML score skews quotes
      - k_fill: fill-probability sensitivity (higher => fewer fills)
    """
    os.makedirs(reports_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Quote construction
    df["skew_adj"] = skew_factor * df["ml_score"] * df["mid"]
    df["bid"] = df["mid"] - 0.5 * df["spread"] - df["skew_adj"]
    df["ask"] = df["mid"] + 0.5 * df["spread"] + df["skew_adj"]
    df["fill_prob"] = np.exp(-k_fill * df["spread"] / df["mid"])

    # State
    cash = 0.0
    option_pos = 0.0
    underlying_pos = 0.0
    pnl_list, option_pos_list, hedge_error_list = [], [], []
    fills = 0

    for _, row in df.iterrows():
        # independent toy probabilities of getting hit on each side
        p_bid = row["fill_prob"] * 0.5
        p_ask = row["fill_prob"] * 0.5
        hit_bid = rng.random() < p_bid
        hit_ask = rng.random() < p_ask
        size = 1.0

        if hit_ask:
            # we sell option at our ask -> short option
            option_pos -= size
            cash += row["ask"] * size
            # delta-hedge: target underlying = - option_pos * delta
            target_u = -option_pos * row["theo_delta"]
            delta_trade = target_u - underlying_pos
            underlying_pos += delta_trade
            cash -= delta_trade * row["underlying"]
            fills += 1

        if hit_bid:
            # we buy option at our bid -> long option
            option_pos += size
            cash -= row["bid"] * size
            target_u = -option_pos * row["theo_delta"]
            delta_trade = target_u - underlying_pos
            underlying_pos += delta_trade
            cash -= delta_trade * row["underlying"]
            fills += 1

        mtm = cash + option_pos * row["mid"] + underlying_pos * row["underlying"]
        pnl_list.append(mtm)
        option_pos_list.append(option_pos)
        hedge_error_list.append(option_pos * row["theo_delta"] + underlying_pos)

    df["pnl_mtm"] = pnl_list
    df["option_pos"] = option_pos_list
    df["hedge_error"] = hedge_error_list

    # Summary
    net_pnl = float(df["pnl_mtm"].iloc[-1])
    rets = pd.Series(df["pnl_mtm"]).diff().fillna(0)
    mean_r = float(rets.mean())
    std_r = float(rets.std() + 1e-12)
    sharpe_per_min = mean_r / std_r if std_r > 0 else 0.0
    annualized_sharpe = sharpe_per_min * math.sqrt(252 * 390)

    summary = {
        "net_pnl": net_pnl,
        "annualized_sharpe_est": float(annualized_sharpe),
        "total_fills": int(fills),
        "avg_inventory": float(abs(df["option_pos"]).mean()),
    }
    print("SUMMARY:", summary)

    # Plots
    plt.figure(figsize=(8, 3.5))
    plt.plot(df["minute"], df["pnl_mtm"])
    plt.title("Cumulative P&L (MTM)")
    plt.xlabel("Minute")
    plt.ylabel("P&L")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "fig_pnl.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 3.5))
    plt.plot(df["minute"], df["option_pos"])
    plt.title("Option Position (inventory) over time")
    plt.xlabel("Minute")
    plt.ylabel("Option contracts")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "fig_inventory.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.hist(df["option_pos"], bins=20)
    plt.title("Inventory distribution (option position)")
    plt.xlabel("Option contracts")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "fig_inventory_hist.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(df["ml_score"], df["parity_residual"], s=10)
    plt.title("Parity residual vs ML score (feature relation)")
    plt.xlabel("ML score (normalized)")
    plt.ylabel("Parity residual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "fig_parity_vs_ml.png"), dpi=150)
    plt.close()

    print(
        "Saved plots to reports/: fig_pnl.png, fig_inventory.png, "
        "fig_inventory_hist.png, fig_parity_vs_ml.png"
    )
    return summary


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="run synthetic demo")
    p.add_argument("--n", type=int, default=800, help="number of synthetic minutes")
    p.add_argument("--skew", type=float, default=0.02, help="quote skew factor")
    p.add_argument("--k-fill", type=float, default=50.0, dest="k_fill",
                   help="fill-probability sensitivity (higher=fewer fills)")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    if args.demo:
        df = build_synthetic(n_minutes=args.n)
        run_engine(
            df,
            reports_dir="reports",
            seed=args.seed,
            skew_factor=args.skew,
            k_fill=args.k_fill,
        )


if __name__ == "__main__":
    main()
