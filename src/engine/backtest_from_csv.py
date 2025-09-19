#!/usr/bin/env python3
# src/engine/backtest_from_csv.py
# NBBO-aware option MM backtest with asymmetric quoting, soft-cap, and conservative adaptive rules.

import argparse, os, glob, math, json, random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- BS utils (no SciPy) ----------------
SQRT2 = math.sqrt(2.0)
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    T = max(T, 1e-8)
    sigma = max(sigma, 1e-8)
    if S <= 0 or K <= 0:
        return 0.0, 0.0
    f = S * math.exp(-q * T)
    df = math.exp(-r * T)
    vol = sigma * math.sqrt(T)
    d1 = (math.log(f / K) + 0.5 * sigma * sigma * T) / vol
    d2 = d1 - vol
    price = df * (f * _norm_cdf(d1) - K * _norm_cdf(d2))
    delta = math.exp(-q * T) * _norm_cdf(d1)  # 0..1 w.r.t. spot
    return price, delta

# ---------------- helpers ----------------
def clamp(x, lo, hi): return max(lo, min(hi, x))

def zscore(x: pd.Series, win: int = 60) -> pd.Series:
    mu = x.rolling(win, min_periods=max(5, win//5)).mean()
    sd = x.rolling(win, min_periods=max(5, win//5)).std(ddof=0)
    z = (x - mu) / (sd.replace(0, np.nan))
    return z.fillna(0.0).clip(-3.5, 3.5)

def realized_vol_ann(ret: pd.Series, win: int = 60) -> pd.Series:
    ann = ret.rolling(win, min_periods=max(5, win//5)).std(ddof=0) * math.sqrt(252.0 * 390.0)
    return ann.bfill().fillna(0.0)

def pct_change_clip(x: pd.Series, max_bps_per_min: float = 200.0) -> pd.Series:
    r = x.pct_change().fillna(0.0)
    cap = max_bps_per_min / 10000.0
    return r.clip(lower=-cap, upper=cap)

# NBBO-aware, single-side hit prob
def hit_prob(side: str, our_px: float, nbbo_bid: float, nbbo_ask: float,
             mid: float, base_at_nbbo: float,
             alpha_inside_bps: float, alpha_outside_bps: float) -> float:
    if mid <= 0 or not np.isfinite(mid): return 0.0
    bp = 1e4
    if side == "bid":
        if our_px >= nbbo_ask: return 0.0
        if our_px > nbbo_bid:
            d_bps = bp * (our_px - nbbo_bid) / mid
            return float(1.0 - math.exp(-alpha_inside_bps * d_bps / bp))
        elif our_px == nbbo_bid:
            return base_at_nbbo
        else:
            d_bps = bp * (nbbo_bid - our_px) / mid
            return float(base_at_nbbo * math.exp(-alpha_outside_bps * d_bps / bp))
    else:
        if our_px <= nbbo_bid: return 0.0
        if our_px < nbbo_ask:
            d_bps = bp * (nbbo_ask - our_px) / mid
            return float(1.0 - math.exp(-alpha_inside_bps * d_bps / bp))
        elif our_px == nbbo_ask:
            return base_at_nbbo
        else:
            d_bps = bp * (our_px - nbbo_ask) / mid
            return float(base_at_nbbo * math.exp(-alpha_outside_bps * d_bps / bp))

def choose_fill(p_buy: float, p_sell: float, rng: random.Random) -> Tuple[int,int]:
    p_buy = clamp(p_buy, 0.0, 1.0); p_sell = clamp(p_sell, 0.0, 1.0)
    tot = p_buy + p_sell
    if tot <= 0.0: return (0,0)
    if tot >= 1.0:
        p_buy /= tot; p_sell /= tot
    u = rng.random()
    if u < p_buy: return (1,0)
    if u < p_buy + p_sell: return (0,1)
    return (0,0)

# ---------------- IO ----------------
def clean_reports(dirpath: str):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True); return
    removed = 0
    for pat in ["fig_*_*.png", "summary_*.json"]:
        for f in glob.glob(os.path.join(dirpath, pat)):
            try: os.remove(f); removed += 1
            except: pass
    if removed:
        print(f"Cleaned {removed} old artifact(s) from '{dirpath}'")

def load_option_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    df["expiry"]    = pd.to_datetime(df["expiry"],    utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["timestamp","expiry"]).reset_index(drop=True)

    for c in ["underlying","strike","iv","r","dividend_yield"]:
        if c not in df.columns: raise ValueError(f"CSV missing column: {c}")

    if "call_bid" not in df.columns or "call_ask" not in df.columns:
        if "call_mid" not in df.columns:
            raise ValueError("Need call_bid/ask or call_mid")
        half = 0.002 * df["call_mid"].values
        df["call_bid"] = df["call_mid"] - half
        df["call_ask"] = df["call_mid"] + half
    if "call_mid" not in df.columns:
        df["call_mid"] = 0.5*(df["call_bid"].values + df["call_ask"].values)

    # sanitize NBBO
    bad = df["call_bid"] > df["call_ask"]
    if bad.any():
        b = df.loc[bad, "call_bid"].copy()
        df.loc[bad, "call_bid"] = df.loc[bad, "call_ask"]
        df.loc[bad, "call_ask"] = b
        df.loc[bad, "call_ask"] = df.loc[bad, "call_bid"] + 1e-6
    df["call_mid"] = 0.5*(df["call_bid"] + df["call_ask"])

    df["T_years"] = ((df["expiry"] - df["timestamp"]).dt.total_seconds() / (252.0*86400.0)).clip(lower=1e-8)

    # sanity underlying and mid
    S_raw = df["underlying"].astype(float)
    rS = pct_change_clip(S_raw, 200.0)
    S_c = S_raw.iloc[0]*(1.0 + rS).cumprod()
    df["underlying_sane"] = S_c.values
    cm = df["call_mid"].astype(float).clip(lower=0.0)
    df["call_mid_sane"] = np.minimum(cm.values, df["underlying_sane"].values)

    df["ret"] = df["underlying_sane"].pct_change().fillna(0.0)
    df["vol_ann"] = realized_vol_ann(df["ret"], 60)
    df["ml_score"] = zscore(df["ret"], 60)

    mid_next = df["call_mid_sane"].shift(-1)
    d_mid_bps = ((mid_next - df["call_mid_sane"]) / df["call_mid_sane"].replace(0, np.nan)) * 1e4
    df["d_mid_bps_next"] = d_mid_bps.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    width = (df["call_ask"] - df["call_bid"]).astype(float)
    df["ok_bar"] = (width <= np.maximum(0.5, 1.0 * df["call_mid_sane"]))

    return df

# ---------------- Adaptive state ----------------
@dataclass
class ParamState:
    k_widen_base: float
    k_widen_pos: float
    k_inv: float
    k_fill: float
    fill_bias: float
    skew: float
    hedge_every: int
    hedge_band_shares: float
    size_contracts: int
    pos_cap: int
    # asym + soft cap
    soft_cap_frac: float
    expo_widen_mult: float
    flat_tight_mult: float

    # conservative clamps
    k_widen_base_min: float = 0.005
    k_widen_base_max: float = 0.02
    k_widen_pos_min:  float = 0.0
    k_widen_pos_max:  float = 0.05
    k_inv_min:        float = 0.0
    k_inv_max:        float = 0.012
    k_fill_min:       float = 35.0
    k_fill_max:       float = 70.0
    fill_bias_min:    float = -0.02
    fill_bias_max:    float = +0.15
    hedge_every_min:  int   = 3
    hedge_every_max:  int   = 10
    hedge_band_min:   float = 80.0
    hedge_band_max:   float = 400.0
    size_min:         int   = 1
    size_max:         int   = 3
    pos_cap_min:      int   = 6

    def clamp_all(self):
        self.k_widen_base = clamp(self.k_widen_base, self.k_widen_base_min, self.k_widen_base_max)
        self.k_widen_pos  = clamp(self.k_widen_pos,  self.k_widen_pos_min,  self.k_widen_pos_max)
        self.k_inv        = clamp(self.k_inv,        self.k_inv_min,        self.k_inv_max)
        self.k_fill       = clamp(self.k_fill,       self.k_fill_min,       self.k_fill_max)
        self.fill_bias    = clamp(self.fill_bias,    self.fill_bias_min,    self.fill_bias_max)
        self.hedge_every  = int(clamp(self.hedge_every, self.hedge_every_min, self.hedge_every_max))
        self.hedge_band_shares = clamp(self.hedge_band_shares, self.hedge_band_min, self.hedge_band_max)
        self.size_contracts = int(clamp(self.size_contracts, self.size_min, self.size_max))
        self.pos_cap = max(self.pos_cap_min, int(self.pos_cap))

# ---------------- Core backtest ----------------
@dataclass
class Args:
    csv: str
    run_id: str
    clean_reports: bool
    contract_mult: float
    opt_fee: float
    hedge_bps: float
    hedge_every: int
    hedge_band_shares: float
    size_contracts: int
    pos_cap: int
    # quoting
    skew: float
    k_widen_base: float
    k_widen_pos: float
    k_inv: float
    k_fill: float
    fill_bias: float
    # fill model
    base_hit_at_nbbo: float
    alpha_inside_bps: float
    alpha_outside_bps: float
    # asym + soft cap
    soft_cap_frac: float
    expo_widen_mult: float
    flat_tight_mult: float
    # adaptivity
    adapt: bool
    adapt_every: int
    win_short: int
    win_long: int
    seed: int

def backtest(df: pd.DataFrame, a: Args) -> dict:
    rng = random.Random(a.seed)
    n = len(df)

    pos_opt = 0         # option position (contracts)
    pos_hedge = 0.0     # shares
    cash_net = 0.0

    total_option_fees = 0.0
    total_hedge_cost  = 0.0
    fills_contracts   = 0
    steps_with_fill   = 0

    pnl_path: List[float] = []
    inv_path: List[int] = []
    pvml: List[Tuple[float,float]] = []

    theo = np.zeros(n); delta = np.zeros(n)
    for i, row in df.iterrows():
        c, d = bs_call(float(row.underlying_sane), float(row.strike), float(row.T_years),
                       float(row.r), float(row.dividend_yield), float(row.iv))
        theo[i] = c; delta[i] = d

    S  = df["underlying_sane"].values
    mid = df["call_mid_sane"].values
    bbo_bid = df["call_bid"].values
    bbo_ask = df["call_ask"].values
    ml = df["ml_score"].values
    d_mid_bps_next = df["d_mid_bps_next"].values
    ok_bar = df["ok_bar"].values

    ps = ParamState(
        k_widen_base=a.k_widen_base, k_widen_pos=a.k_widen_pos, k_inv=a.k_inv,
        k_fill=a.k_fill, fill_bias=a.fill_bias, skew=a.skew,
        hedge_every=a.hedge_every, hedge_band_shares=a.hedge_band_shares,
        size_contracts=a.size_contracts, pos_cap=a.pos_cap,
        soft_cap_frac=a.soft_cap_frac, expo_widen_mult=a.expo_widen_mult, flat_tight_mult=a.flat_tight_mult
    )

    window_short = max(20, a.win_short)
    window_long  = max(window_short+20, a.win_long)
    fill_side_hist: List[int] = []   # +1 buy at bid, -1 sell at ask, 0 none
    adverse_hist: List[float] = []   # bps of next mid vs signed fill
    hedge_cost_hist: List[float] = []
    pnl_peak = -1e99
    dd_hist: List[float] = []
    adapt_cooldown = 0

    for i in range(n):
        if not ok_bar[i] or not np.isfinite(mid[i]) or not np.isfinite(S[i]):
            m = mid[i] if np.isfinite(mid[i]) else (mid[i-1] if i>0 else 0.0)
            s = S[i] if np.isfinite(S[i]) else (S[i-1] if i>0 else 0.0)
            mtm = pos_opt * m * a.contract_mult + pos_hedge * s
            pnl_path.append(cash_net + mtm)
            inv_path.append(pos_opt)
            pvml.append((ml[i] if np.isfinite(ml[i]) else 0.0, (theo[i]-m) if np.isfinite(m) else 0.0))
            pnl_peak = max(pnl_peak, pnl_path[-1])
            dd_hist.append(pnl_path[-1] - pnl_peak)
            hedge_cost_hist.append(0.0)
            continue

        m = float(mid[i]); s = float(S[i])

        # --- Asymmetric quoting + inventory lean ---
        fair = theo[i] + ps.skew * ml[i] * max(0.01, m)
        inv_frac = abs(pos_opt) / max(1, ps.pos_cap)
        base_half = ps.k_widen_base * max(0.01, m) + ps.k_widen_pos * inv_frac * max(0.01, m)
        base_half = max(0.0005, base_half)

        inv_penalty = ps.k_inv * pos_opt
        q_mid = fair - inv_penalty

        # Which side increases exposure?
        # If long, buying more increases exposure; selling reduces. If short, reverse.
        long = pos_opt >= 0
        bid_expo = long      # posting bid (we buy) increases exposure if we're already long
        ask_expo = not long  # posting ask (we sell) increases exposure if we're short

        half_bid = base_half * (ps.expo_widen_mult if bid_expo else ps.flat_tight_mult)
        half_ask = base_half * (ps.expo_widen_mult if ask_expo else ps.flat_tight_mult)

        q_bid = min(q_mid - half_bid, bbo_ask[i] - 1e-6)
        q_ask = max(q_mid + half_ask, bbo_bid[i] + 1e-6)

        # --- Soft cap logic: stop quoting exposure-increasing side above soft threshold ---
        soft_cap = int(math.floor(ps.soft_cap_frac * ps.pos_cap))
        stop_buy = stop_sell = False
        if abs(pos_opt) >= soft_cap:
            if pos_opt >= soft_cap:
                # long: disable buys (exposure-increasing), allow sells
                stop_buy = True
            if -pos_opt >= soft_cap:
                # short: disable sells (exposure-increasing), allow buys
                stop_sell = True

        # Exclusive fill
        p_buy  = 0.0 if stop_buy else hit_prob("bid", q_bid, bbo_bid[i], bbo_ask[i], m,
                                               a.base_hit_at_nbbo, a.alpha_inside_bps, a.alpha_outside_bps)
        p_sell = 0.0 if stop_sell else hit_prob("ask", q_ask, bbo_bid[i], bbo_ask[i], m,
                                                a.base_hit_at_nbbo, a.alpha_inside_bps, a.alpha_outside_bps)

        # Hard cap block
        if pos_opt >= ps.pos_cap:  p_buy = 0.0
        if -pos_opt >= ps.pos_cap: p_sell = 0.0

        do_buy, do_sell = choose_fill(p_buy, p_sell, rng)
        buy_qty  = ps.size_contracts if do_buy  else 0
        sell_qty = ps.size_contracts if do_sell else 0
        any_fill = (buy_qty + sell_qty) > 0
        if any_fill: steps_with_fill += 1

        # Options exec + fees + spread capture (for diagnostics)
        if buy_qty:
            exe = q_bid
            cash_net -= exe * buy_qty * a.contract_mult
            fee = a.opt_fee * buy_qty
            cash_net -= fee; total_option_fees += fee
            pos_opt += buy_qty; fills_contracts += buy_qty
        if sell_qty:
            exe = q_ask
            cash_net += exe * sell_qty * a.contract_mult
            fee = a.opt_fee * sell_qty
            cash_net -= fee; total_option_fees += fee
            pos_opt -= sell_qty; fills_contracts += sell_qty

        pos_opt = int(clamp(pos_opt, -ps.pos_cap, ps.pos_cap))

        signed = (1 if buy_qty else 0) - (1 if sell_qty else 0)
        adverse_hist.append(signed * d_mid_bps_next[i])
        fill_side_hist.append(signed)

        # Hedge (band + cadence)
        opt_delta_sh = pos_opt * delta[i] * a.contract_mult
        delta_err = opt_delta_sh + pos_hedge
        fee_h = 0.0
        if (i % max(1, ps.hedge_every)) == 0 and abs(delta_err) > ps.hedge_band_shares:
            trade = -math.copysign(min(abs(delta_err), ps.hedge_band_shares), delta_err)
            fee_h = abs(trade) * s * (a.hedge_bps * 1e-4)
            cash_net -= trade * s
            cash_net -= fee_h
            pos_hedge += trade
            total_hedge_cost += fee_h
        hedge_cost_hist.append(fee_h)

        # Mark-to-market
        mtm = pos_opt * m * a.contract_mult + pos_hedge * s
        net_now = cash_net + mtm
        pnl_path.append(net_now)
        inv_path.append(pos_opt)
        pvml.append((ml[i], theo[i] - m))
        pnl_peak = max(pnl_peak, net_now)
        dd_hist.append(net_now - pnl_peak)

        # -------------- Adaptive loop (conservative) --------------
        if a.adapt and i >= window_long and (i % max(1, a.adapt_every) == 0) and adapt_cooldown == 0:
            inv_abs_mean = float(np.mean(np.abs(inv_path[-window_long:]))) / max(1, ps.pos_cap)
            fill_rate_long = float(np.mean([1 if x != 0 else 0 for x in fill_side_hist[-window_long:]]))
            adverse_mean_bps = float(np.mean(adverse_hist[-window_short:]))   # want >= 0
            hedge_cost_pm = float(np.mean(hedge_cost_hist[-window_short:]))   # $/bar
            dd_now = float(dd_hist[-1])

            # vol regime
            ret_w = df["ret"].iloc[max(0, i-window_long):i]
            if len(ret_w) >= 40:
                vol_s = ret_w.rolling(60, min_periods=20).std(ddof=0).iloc[-1] * math.sqrt(252.0*390.0)
                vol_l = ret_w.rolling(240, min_periods=40).std(ddof=0).iloc[-1] * math.sqrt(252.0*390.0)
                vol_ratio = (vol_s / (vol_l + 1e-12)) if vol_l > 0 else 1.0
            else:
                vol_ratio = 1.0

            changed = False

            # R1: inventory high -> push flattening preference (no base reduce allowed)
            if inv_abs_mean > 0.65:
                ps.k_inv        = clamp(ps.k_inv + 0.002, ps.k_inv_min, ps.k_inv_max)
                ps.k_widen_pos  = clamp(ps.k_widen_pos + 0.005, ps.k_widen_pos_min, ps.k_widen_pos_max)
                ps.fill_bias    = clamp(ps.fill_bias - 0.02, ps.fill_bias_min, ps.fill_bias_max)
                ps.size_contracts = max(ps.size_min, ps.size_contracts - 1)
                # ensure a healthy base width when inventory stressed
                ps.k_widen_base = max(ps.k_widen_base, 0.008)
                changed = True

            # R2: vol spike
            if vol_ratio > 1.4:
                ps.k_widen_base = clamp(ps.k_widen_base + 0.002, ps.k_widen_base_min, ps.k_widen_base_max)
                ps.k_fill       = clamp(ps.k_fill - 3, ps.k_fill_min, ps.k_fill_max)
                ps.hedge_every  = max(ps.hedge_every_min, ps.hedge_every - 1)
                ps.hedge_band_shares = max(ps.hedge_band_min, ps.hedge_band_shares * 0.9)
                ps.skew         = max(0.0, ps.skew * 0.85)
                changed = True

            # R3: fill-rate too high
            if fill_rate_long > 0.30:
                ps.k_widen_base = clamp(ps.k_widen_base + 0.002, ps.k_widen_base_min, ps.k_widen_base_max)
                ps.fill_bias    = clamp(ps.fill_bias - 0.02, ps.fill_bias_min, ps.fill_bias_max)
                ps.k_fill       = clamp(ps.k_fill - 3, ps.k_fill_min, ps.k_fill_max)
                changed = True

            # R4: fill-rate too low (but don't reduce base if inv stressed)
            if fill_rate_long < 0.06 and inv_abs_mean < 0.50:
                ps.k_widen_base = clamp(ps.k_widen_base - 0.001, ps.k_widen_base_min, ps.k_widen_base_max)
                ps.fill_bias    = clamp(ps.fill_bias + 0.02, ps.fill_bias_min, ps.fill_bias_max)
                ps.k_fill       = clamp(ps.k_fill + 2, ps.k_fill_min, ps.k_fill_max)
                changed = True

            # R5: adverse selection
            if adverse_mean_bps < -0.5:
                ps.k_fill       = clamp(ps.k_fill - 3, ps.k_fill_min, ps.k_fill_max)
                ps.fill_bias    = clamp(ps.fill_bias - 0.02, ps.fill_bias_min, ps.fill_bias_max)
                ps.k_widen_base = clamp(ps.k_widen_base + 0.001, ps.k_widen_base_min, ps.k_widen_base_max)
                ps.skew         = max(0.0, ps.skew * 0.9)
                changed = True

            # R6: hedge cost high
            if hedge_cost_pm > 5.0:
                ps.hedge_every  = clamp(ps.hedge_every + 1, ps.hedge_every_min, ps.hedge_every_max)
                ps.hedge_band_shares = clamp(ps.hedge_band_shares * 1.15, ps.hedge_band_min, ps.hedge_band_max)
                changed = True

            # R7: drawdown guard (gentle)
            if dd_now < -2000.0:
                ps.pos_cap      = max(ps.pos_cap_min, int(ps.pos_cap * 0.9))
                ps.k_widen_base = clamp(ps.k_widen_base + 0.001, ps.k_widen_base_min, ps.k_widen_base_max)
                ps.k_inv        = clamp(ps.k_inv + 0.001, ps.k_inv_min, ps.k_inv_max)
                ps.fill_bias    = clamp(ps.fill_bias - 0.01, ps.fill_bias_min, ps.fill_bias_max)
                ps.skew         = max(0.0, ps.skew * 0.9)
                changed = True

            if changed:
                ps.clamp_all()
                adapt_cooldown = a.adapt_every

        adapt_cooldown = max(0, adapt_cooldown - 1)

    # summary
    pnl_path = np.asarray(pnl_path, dtype=float)
    inv_path = np.asarray(inv_path, dtype=float)
    ret = np.diff(np.concatenate([[pnl_path[0] if len(pnl_path) else 0.0], pnl_path]))
    st = ret.std(ddof=0)
    sharpe = float((ret.mean() / (st + 1e-12)) * math.sqrt(252.0*390.0)) if len(ret) else 0.0
    max_dd = float(np.min(pnl_path - np.maximum.accumulate(pnl_path))) if len(pnl_path) else 0.0

    summary = dict(
        net_pnl=float(pnl_path[-1]) if len(pnl_path) else 0.0,
        annualized_sharpe_est=sharpe,
        total_fills=int(fills_contracts),
        fill_rate=float(steps_with_fill / max(1, n)),
        avg_inventory_contracts=float(np.mean(np.abs(inv_path))) if len(inv_path) else 0.0,
        hedge_error_rms=0.0,
        max_drawdown=max_dd,
        contract_mult=float(a.contract_mult),
        opt_fee_per_contract=float(a.opt_fee),
        hedge_fee_bps=float(a.hedge_bps),
        total_option_fees=float(total_option_fees),
        total_hedge_cost=float(total_hedge_cost),
        gross_pnl=float((pnl_path[-1] if len(pnl_path) else 0.0) + total_option_fees + total_hedge_cost),
        final_params=dict(
            k_widen_base=ps.k_widen_base, k_widen_pos=ps.k_widen_pos, k_inv=ps.k_inv,
            k_fill=ps.k_fill, fill_bias=ps.fill_bias, skew=ps.skew,
            hedge_every=ps.hedge_every, hedge_band_shares=ps.hedge_band_shares,
            size_contracts=ps.size_contracts, pos_cap=ps.pos_cap,
            soft_cap_frac=ps.soft_cap_frac, expo_widen_mult=ps.expo_widen_mult, flat_tight_mult=ps.flat_tight_mult
        ),
    )
    return summary, pnl_path, inv_path, pvml, total_option_fees, total_hedge_cost

# ---------------- plots ----------------
def out_name(base: str, run_id: str) -> str:
    return f"reports/{base}{'' if not run_id else '_' + run_id}.png"

def plot_all(pnl_path, inv_path, pvml, run_id: str, fee_opt: float, cost_hedge: float):
    plt.figure(figsize=(11,4))
    plt.plot(pnl_path + (fee_opt + cost_hedge), label="Gross P&L")
    plt.plot(pnl_path, label="Net P&L (after costs)")
    plt.title("Cumulative P&L — Gross vs Net"); plt.xlabel("Minute"); plt.ylabel("P&L")
    plt.legend(); plt.tight_layout(); plt.savefig(out_name("fig_pnl", run_id), dpi=150); plt.close()

    plt.figure(figsize=(11,3.2))
    plt.plot(inv_path)
    plt.title("Option Position (inventory) over time")
    plt.xlabel("Minute"); plt.ylabel("Option contracts")
    plt.tight_layout(); plt.savefig(out_name("fig_inventory", run_id), dpi=150); plt.close()

    plt.figure(figsize=(9,4))
    plt.hist(inv_path, bins=30)
    plt.title("Inventory distribution (option position)")
    plt.xlabel("Option contracts"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out_name("fig_inventory_hist", run_id), dpi=150); plt.close()

    xs = [x for (x, y) in pvml]; ys = [y for (x, y) in pvml]
    plt.figure(figsize=(7.5,4.8))
    plt.scatter(xs, ys, s=10)
    plt.title("Parity residual (theo - mid) vs ML score")
    plt.xlabel("ML score (z)"); plt.ylabel("theo - mid")
    plt.tight_layout(); plt.savefig(out_name("fig_parity_vs_ml", run_id), dpi=150); plt.close()

# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Adaptive NBBO-aware option MM (single series) with asymmetric quoting + soft-cap.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--clean-reports", action="store_true")

    # trading
    ap.add_argument("--contract-mult", type=float, default=100.0)
    ap.add_argument("--opt-fee", type=float, default=0.65)
    ap.add_argument("--hedge-bps", type=float, default=2.0)
    ap.add_argument("--hedge-every", type=int, default=5)
    ap.add_argument("--hedge-band-shares", type=float, default=150.0)
    ap.add_argument("--size-contracts", type=int, default=1)
    ap.add_argument("--pos-cap", type=int, default=12)

    # quoting base
    ap.add_argument("--skew", type=float, default=0.02)
    ap.add_argument("--k-widen-base", type=float, default=0.008)
    ap.add_argument("--k-widen-pos", type=float, default=0.02)
    ap.add_argument("--k-inv", type=float, default=0.008)
    ap.add_argument("--k-fill", type=float, default=50.0)
    ap.add_argument("--fill-bias", type=float, default=0.10)

    # fill model shape
    ap.add_argument("--base-hit-at-nbbo", type=float, default=0.03)
    ap.add_argument("--alpha-inside-bps", type=float, default=80.0)
    ap.add_argument("--alpha-outside-bps", type=float, default=180.0)

    # asym + soft cap
    ap.add_argument("--soft-cap-frac", type=float, default=0.75)
    ap.add_argument("--expo-widen-mult", type=float, default=2.0)
    ap.add_argument("--flat-tight-mult", type=float, default=0.8)

    # adaptivity
    ap.add_argument("--adapt", action="store_true")
    ap.add_argument("--adapt-every", type=int, default=8)
    ap.add_argument("--win-short", type=int, default=30)
    ap.add_argument("--win-long", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()

def main():
    a = parse_args()
    if a.clean_reports: clean_reports("reports")
    df = load_option_csv(a.csv)

    # pack args into dataclass-like object
    summary, pnl_path, inv_path, pvml, fee_opt, cost_hedge = backtest(
        df,
        Args(
            csv=a.csv, run_id=a.run_id, clean_reports=a.clean_reports,
            contract_mult=a.contract_mult, opt_fee=a.opt_fee, hedge_bps=a.hedge_bps,
            hedge_every=a.hedge_every, hedge_band_shares=a.hedge_band_shares,
            size_contracts=a.size_contracts, pos_cap=a.pos_cap,
            skew=a.skew, k_widen_base=a.k_widen_base, k_widen_pos=a.k_widen_pos,
            k_inv=a.k_inv, k_fill=a.k_fill, fill_bias=a.fill_bias,
            base_hit_at_nbbo=a.base_hit_at_nbbo, alpha_inside_bps=a.alpha_inside_bps, alpha_outside_bps=a.alpha_outside_bps,
            soft_cap_frac=a.soft_cap_frac, expo_widen_mult=a.expo_widen_mult, flat_tight_mult=a.flat_tight_mult,
            adapt=a.adapt, adapt_every=a.adapt_every, win_short=a.win_short, win_long=a.win_long, seed=a.seed
        )
    )

    plot_all(pnl_path, inv_path, pvml, a.run_id, fee_opt, cost_hedge)

    print("Saved plots to reports/:",
          f"fig_pnl_{a.run_id}.png, fig_inventory_{a.run_id}.png,",
          f"fig_inventory_hist_{a.run_id}.png, fig_parity_vs_ml_{a.run_id}.png")
    print("SUMMARY:", summary)
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/summary_{a.run_id}.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
