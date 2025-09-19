#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic minute-level demo with the same mechanics as the real-data engine:
- Generates an underlier path + a single option series (strike near ATM).
- Risk-aware quoting (inventory skew), simple regime selection.
- Realistic fills (competition/adverse/queue), partial fills.
- Delta hedging with band & costs.
- Saves plots in reports/.
"""

from __future__ import annotations
import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def ensure_reports(clean: bool) -> None:
    os.makedirs("reports", exist_ok=True)
    if clean:
        removed = 0
        for p in glob.glob("reports/fig_*.png"):
            try:
                os.remove(p)
                removed += 1
            except Exception:
                pass
        if removed:
            print(f"Cleaned {removed} old artifact(s) from 'reports'")


def bs_call_delta(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return float(np.exp(-q * T) * norm.cdf(d1))


def bs_call_price(S, K, T, r, q, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def synthetic_df(n=390, S0=450.0, mu=0.0, vol=0.18, seed=7):
    rng = np.random.default_rng(seed)
    dt = 1.0 / (252.0 * 390.0)
    rets = rng.normal(mu * dt, vol * np.sqrt(dt), size=n)
    S = S0 * np.exp(np.cumsum(rets))
    t0 = pd.Timestamp("2024-07-01 09:30:00")
    ts = pd.date_range(t0, periods=n, freq="1min")

    # ATM-ish strike, ~30d expiry
    K0 = round(S0 / 5) * 5.0
    K = np.full(n, K0)
    expiry = pd.Timestamp("2024-08-01 00:00:00")

    df = pd.DataFrame(
        dict(
            timestamp=ts,
            underlying=S,
            strike=K,
            expiry=[expiry] * n,
            r=0.01,
            iv=0.22,
            dividend_yield=0.012,
        )
    )
    # Option mids (use BS)
    T_years = ((df["expiry"] - df["timestamp"]).dt.total_seconds()) / (252 * 86400)
    df["T_years"] = T_years.clip(lower=0.0)
    df["call_mid"] = [
        bs_call_price(S=float(s), K=float(k), T=float(t), r=0.01, q=0.012, sigma=0.22)
        for s, k, t in zip(df["underlying"], df["strike"], df["T_years"])
    ]
    # Observed spread proxy
    df["nbbo_w"] = 0.04 + 0.02 * (np.abs(pd.Series(rets)).rolling(5).mean().fillna(0.0).values)

    # Features (past only)
    df["ret1"] = pd.Series(S).pct_change()
    df["ret1_lag"] = df["ret1"].shift(1)
    df["oi"] = df["ret1_lag"].rolling(5, min_periods=5).sum().clip(-0.02, 0.02).fillna(0.0)
    m = df["oi"].rolling(100, min_periods=20).mean().shift(1)
    s = df["oi"].rolling(100, min_periods=20).std(ddof=0).shift(1)
    df["ml_score"] = ((df["oi"] - m) / s.replace(0, np.nan)).fillna(0.0).clip(-3, 3)
    df["dS_next"] = df["underlying"].diff().shift(-1).fillna(0.0)
    return df.reset_index(drop=True)


def save_plots(tag, minute, pnl_gross, pnl_net, inv_ts):
    suffix = f"_{tag}" if tag else ""
    plt.figure(figsize=(11, 4))
    plt.plot(minute, pnl_gross, label="Gross P&L")
    plt.plot(minute, pnl_net, label="Net P&L (after costs)")
    plt.title("Cumulative P&L — Gross vs Net")
    plt.xlabel("Minute")
    plt.ylabel("P&L")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/fig_pnl{suffix}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 3.4))
    plt.plot(minute, inv_ts)
    plt.title("Option Position (inventory) over time")
    plt.xlabel("Minute")
    plt.ylabel("Option contracts")
    plt.tight_layout()
    plt.savefig(f"reports/fig_inventory{suffix}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.hist(inv_ts, bins=20)
    plt.title("Inventory distribution (option position)")
    plt.xlabel("Option contracts")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"reports/fig_inventory_hist{suffix}.png", dpi=160)
    plt.close()


def backtest(df, args):
    rng = np.random.default_rng(args.seed)
    inv = 0
    shares = 0.0
    cash_gross = 0.0
    cost_accum = 0.0
    pnl_g, pnl_n, inv_ts, minute = [], [], [], []

    for i, row in df.iterrows():
        S = float(row["underlying"])
        K = float(row["strike"])
        T = float(row["T_years"])
        call_mid = float(row["call_mid"])
        sigma = float(row["iv"])

        observed_w = float(row["nbbo_w"])
        half_spread = max(observed_w * 0.5 * (1.0 + args.k_widen_base), args.opt_tick * 0.5)

        # Simple regime selection
        inv_frac = abs(inv) / max(1, args.pos_cap)
        k_widen = 0.02
        k_inv = 0.004
        k_fill = 60.0
        hedge_band = 100.0
        if inv_frac > 0.7:
            k_inv = 0.010
            k_widen = max(k_widen, 0.03)
            hedge_band = 60.0
        if args.k_widen is not None: k_widen = args.k_widen
        if args.k_inv is not None:   k_inv = args.k_inv
        if args.k_fill is not None:  k_fill = args.k_fill
        hedge_band = args.hedge_band_shares or hedge_band

        block_buy = inv >= args.pos_cap
        block_sell = inv <= -args.pos_cap

        ml_score = float(row.get("ml_score", 0.0))
        skew_dollars = (args.skew * ml_score + k_inv * inv)

        bid = call_mid - half_spread - skew_dollars
        ask = call_mid + half_spread - skew_dollars

        best_bid = call_mid - observed_w * 0.5
        best_ask = call_mid + observed_w * 0.5
        edge_bid = (best_bid - bid) / args.opt_tick
        edge_ask = (ask - best_ask) / args.opt_tick

        oi = float(row.get("oi", 0.0))
        next_move = float(row.get("dS_next", 0.0))

        def fillprob(side, edge_ticks):
            adverse = (side == "buy" and next_move < 0) or (side == "sell" and next_move > 0)
            p_base = sigmoid(-1.0 - 1.2 * edge_ticks + 8.0 * oi)
            p_adv = 1.6 if adverse else 1.0
            p_queue = np.exp(-10.0 / 120.0) * 0.9
            p = p_base * p_adv * p_queue
            return float(np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0))

        p_buy = 0.0 if block_buy else fillprob("buy", edge_bid) * (1.0 + args.fill_bias)
        p_sell = 0.0 if block_sell else fillprob("sell", edge_ask) * (1.0 - args.fill_bias)
        p_buy = float(np.clip(np.nan_to_num(p_buy, nan=0.0), 0.0, 1.0))
        p_sell = float(np.clip(np.nan_to_num(p_sell, nan=0.0), 0.0, 1.0))

        buy_qty = int(rng.binomial(args.size_contracts, p_buy))
        sell_qty = int(rng.binomial(args.size_contracts, p_sell))

        if buy_qty > 0:
            price = bid + args.opt_slip_ticks * args.opt_tick
            cash_gross -= price * buy_qty * args.contract_mult
            inv += buy_qty
            cost_accum += args.opt_fee * buy_qty

        if sell_qty > 0:
            price = ask - args.opt_slip_ticks * args.opt_tick
            cash_gross += price * sell_qty * args.contract_mult
            inv -= sell_qty
            cost_accum += args.opt_fee * sell_qty

        if args.hedge_every and (i % args.hedge_every == 0):
            delta = bs_call_delta(S, K, T, r=0.01, q=0.012, sigma=sigma)
            target = -inv * delta * args.contract_mult
            gap = target - shares
            if abs(gap) > hedge_band:
                trade = gap - np.sign(gap) * hedge_band
                slip_bps = args.hedge_slip_bps + args.hedge_bps
                px = S * (1.0 + np.sign(trade) * slip_bps * 1e-4)
                cash_gross -= px * trade
                shares += trade
                cost_accum += abs(trade) * S * (args.hedge_bps * 1e-4)

        port_g = cash_gross + inv * call_mid * args.contract_mult + shares * S
        pnl_g.append(port_g)
        pnl_n.append(port_g - cost_accum)
        inv_ts.append(inv)
        minute.append(i)

    pnl_g = np.array(pnl_g)
    pnl_n = np.array(pnl_n)
    ret = np.diff(pnl_n, prepend=pnl_n[0])
    sharpe = (ret.mean() / (ret.std(ddof=1) + 1e-12)) * np.sqrt(252 * 390)
    dd = pnl_n - np.maximum.accumulate(pnl_n)
    max_dd = float(dd.min())  # not printed but available if desired

    save_plots(args.run_id, np.array(minute), pnl_g, pnl_n, np.array(inv_ts))

    return dict(
        net_pnl=float(pnl_n[-1]),
        annualized_sharpe_est=float(sharpe),
        total_fills=int(np.sum(np.array(inv_ts[:-1]) != np.array(inv_ts[1:]))),
        avg_inventory=float(np.mean(inv_ts)),
    )


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--n", type=int, default=390)
    p.add_argument("--skew", type=float, default=0.02)
    p.add_argument("--k-inv", type=float, default=None)
    p.add_argument("--k-widen", type=float, default=None)
    p.add_argument("--k-widen-base", type=float, default=0.0)
    p.add_argument("--k-fill", type=float, default=None)
    p.add_argument("--fill-bias", type=float, default=0.0)
    p.add_argument("--pos-cap", type=int, default=12)

    p.add_argument("--contract-mult", type=float, default=100.0)
    p.add_argument("--opt-fee", type=float, default=2.0)
    p.add_argument("--hedge-bps", type=float, default=10.0)
    p.add_argument("--hedge-slip-bps", type=float, default=0.5)
    p.add_argument("--opt-slip-ticks", type=float, default=0.5)
    p.add_argument("--opt-tick", type=float, default=0.01)
    p.add_argument("--under-tick", type=float, default=0.01)

    p.add_argument("--hedge-band-shares", type=float, default=100.0)
    p.add_argument("--hedge-every", type=int, default=5)
    p.add_argument("--size-contracts", type=int, default=1)

    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--clean-reports", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    return p


def main():
    args = build_arg_parser().parse_args()
    ensure_reports(args.clean_reports)
    df = synthetic_df(n=args.n, seed=args.seed)
    summary = backtest(df, args)
    print("SUMMARY:", summary)
    print("Saved plots to reports/: fig_pnl.png, fig_inventory.png, fig_inventory_hist.png")


if __name__ == "__main__":
    main()
