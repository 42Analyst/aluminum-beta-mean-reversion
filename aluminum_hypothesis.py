# =============================================================================
# Hypothesis Validation v2 — Aluminum as a Cost Shock to Construction Stocks
#
# CORRECTED ECONOMIC MODEL
# ========================
# Aluminum is an INPUT COST for construction (siding, wiring, window frames,
# HVAC ducting). Rising prices squeeze builder margins, especially when costs
# can't be passed on (high mortgage rates, affordability constraints).
# This creates a NEGATIVE / INVERSE relationship — not positive co-movement.
#
# Three hypotheses tested in parallel:
#
#   H1 — INVERSE BETA (corrected)
#        spread = −aluminum − β×stock
#        Rising Al → falling stocks. Test for negative Spearman r and
#        cointegration on the inverse spread.
#
#   H2 — COST-MARGIN RATIO
#        ratio = aluminum_price / stock_price
#        When Al is expensive relative to builder equity, margins are compressed.
#        Mean-reversion in the ratio captures the cost pass-through cycle.
#        Enter: long stock / short Al when ratio is high (Al expensive).
#        Exit: when ratio mean-reverts.
#
#   H3 — DEMAND-DRIVEN POSITIVE BETA (original — kept as null benchmark)
#        spread = aluminum − β×stock
#        Demand booms lift both (more building = more aluminum needed).
#        Expected to score LOW. Retained so you can see the contrast.
#
# Scoring: each hypothesis is scored 0–5 across:
#   1. Directional correlation (sign-checked)
#   2. Granger causality (Al leads stock)
#   3. Cointegration / ratio stationarity
#   4. Spread quality (half-life + Hurst)
#   5. Regime stability (rolling ADF)
#
# Output:
#   hypothesis_report.txt  — structured verdict table, paste into README
#   hypothesis_plots.png   — scatter, spread, ratio, rolling ADF per pair
#
# Usage:
#   python hypothesis_validator.py
#   python hypothesis_validator.py --al AA --stocks XHB ITB PKB VMC
#   python hypothesis_validator.py --start 2018-01-01
# =============================================================================

from __future__ import annotations

import argparse
import textwrap
import warnings
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")

ALPHA = 0.05


# =============================================================================
# DATA
# =============================================================================

def fetch(al_ticker: str, stock_tickers: list[str],
          start: str = "2015-01-01") -> pd.DataFrame:
    end     = date.today().isoformat()
    tickers = [al_ticker] + stock_tickers
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={al_ticker: "aluminum"}).dropna()
    print(f"  {len(raw)} bars  "
          f"({raw.index[0].date()} → {raw.index[-1].date()})")
    return raw


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def test_normality(series: pd.Series) -> dict:
    clean  = series.dropna()
    sw_p   = stats.shapiro(clean[:5000])[1]
    jb_p   = stats.jarque_bera(clean)[1]
    dp_p   = stats.normaltest(clean)[1]
    non_n  = any(p < ALPHA for p in [sw_p, jb_p, dp_p])
    return {"shapiro_p": round(sw_p, 4), "jarque_bera_p": round(jb_p, 4),
            "skewness": round(float(clean.skew()), 3),
            "excess_kurt": round(float(clean.kurtosis()), 3),
            "non_normal": non_n,
            "test_choice": "Spearman" if non_n else "Pearson"}


def test_correlation(al_ret: pd.Series, stk_ret: pd.Series,
                     norm_al: dict, norm_stk: dict,
                     expected_sign: int = 1) -> dict:
    both_n = not norm_al["non_normal"] and not norm_stk["non_normal"]
    if both_n:
        r, p   = stats.pearsonr(al_ret, stk_ret);  method = "Pearson"
    else:
        r, p   = stats.spearmanr(al_ret, stk_ret); method = "Spearman"
    rng   = np.random.default_rng(42)
    boots = [stats.spearmanr(al_ret.iloc[i], stk_ret.iloc[i]).correlation
             for i in (rng.integers(0, len(al_ret), len(al_ret))
                       for _ in range(500))]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    strength     = ("strong" if abs(r) > 0.6 else
                    "moderate" if abs(r) > 0.3 else "weak")
    sign_ok      = (r * expected_sign) > 0
    return {"method": method, "r": round(r, 4), "p": round(p, 6),
            "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "significant": p < ALPHA, "strength": strength,
            "sign_correct": sign_ok,
            "passes": p < ALPHA and strength != "weak" and sign_ok}


def test_stationarity(series: pd.Series, label: str = "") -> dict:
    clean  = series.dropna()
    adf_p  = adfuller(clean, autolag="AIC")[1]
    kpss_p = kpss(clean, regression="c", nlags="auto")[1]
    is_i1  = (adf_p > ALPHA) and (kpss_p < ALPHA)
    return {"label": label, "adf_p": round(adf_p, 4),
            "kpss_p": round(kpss_p, 4), "is_i1": is_i1}


def test_granger(returns: pd.DataFrame, stock_col: str,
                 max_lag: int = 5) -> dict:
    data    = returns[["aluminum", stock_col]].dropna()
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    best_lag, best_p = min(
        ((lag, res[0]["ssr_ftest"][1]) for lag, res in results.items()),
        key=lambda x: x[1],
    )
    return {"best_lag": best_lag, "best_p": round(best_p, 6),
            "passes": best_p < ALPHA}


def test_cointegration(s1: pd.Series, s2: pd.Series) -> dict:
    eg_stat, eg_p, _ = coint(s1, s2)
    mat      = pd.concat([s1, s2], axis=1).dropna().values
    joh      = coint_johansen(mat, det_order=0, k_ar_diff=1)
    tr_stat  = joh.lr1[0]
    tr_crit  = joh.cvt[0, 1]
    joh_pass = tr_stat > tr_crit
    return {"eg_p": round(eg_p, 4), "eg_pass": eg_p < ALPHA,
            "joh_trace": round(tr_stat, 4), "joh_crit": round(tr_crit, 4),
            "joh_pass": joh_pass, "passes": (eg_p < ALPHA) and joh_pass}


def test_spread_quality(spread: pd.Series) -> dict:
    clean = spread.dropna()
    lag   = clean.shift(1).dropna()
    delta = clean.diff().dropna()
    idx   = lag.index.intersection(delta.index)
    m     = LinearRegression().fit(lag.loc[idx].values.reshape(-1, 1),
                                   delta.loc[idx].values)
    lam       = m.coef_[0]
    half_life = -np.log(2) / lam if lam < 0 else np.inf

    def hurst(ts):
        ts  = np.array(ts)
        lgs = range(2, min(100, len(ts) // 2))
        rv  = []
        for l in lgs:
            chs = [ts[i:i + l] for i in range(0, len(ts) - l, l)]
            rc  = [((lambda d: d.max() - d.min())(np.cumsum(c - c.mean()))) / (c.std(ddof=1) or 1)
                   for c in chs if c.std(ddof=1) > 0]
            if rc:
                rv.append((l, np.mean(rc)))
        if len(rv) < 2:
            return 0.5
        return float(np.polyfit(np.log([v[0] for v in rv]),
                                np.log([v[1] for v in rv]), 1)[0])

    h     = hurst(clean.values)
    adf_p = adfuller(clean)[1]
    v     = ("TRADEABLE"     if 5 <= half_life <= 60 and h < 0.5 and adf_p < ALPHA
             else "MARGINAL" if half_life <= 120 and adf_p < 0.10
             else "NOT TRADEABLE")
    return {"half_life_days": round(half_life, 1), "hurst_exp": round(h, 4),
            "spread_adf_p": round(adf_p, 4), "stationary": adf_p < ALPHA,
            "verdict": v, "passes": v == "TRADEABLE"}


def test_regime_stability(spread: pd.Series, window: int = 126) -> dict:
    clean = spread.dropna()
    rp    = pd.Series(
        [adfuller(clean.iloc[i - window:i])[1]
         for i in range(window, len(clean))],
        index=clean.index[window:],
    )
    pct = float((rp > 0.10).mean())
    return {"pct_broken": round(pct * 100, 1),
            "max_adf_p":  round(float(rp.max()), 4),
            "passes":     pct < 0.20,
            "rolling_adf": rp}


# =============================================================================
# SPREAD CONSTRUCTORS
# =============================================================================

def build_spread_rolling(s1: pd.Series, s2: pd.Series,
                         beta_window: int = 252,
                         inverse: bool = False) -> pd.Series:
    betas = [np.nan] * beta_window
    for i in range(beta_window, len(s1)):
        x  = s2.iloc[i - beta_window:i].values.reshape(-1, 1)
        y  = ((-s1) if inverse else s1).iloc[i - beta_window:i].values
        betas.append(LinearRegression().fit(x, y).coef_[0])
    beta_s = pd.Series(betas, index=s1.index)
    al_leg = -s1 if inverse else s1
    return al_leg - beta_s * s2


# =============================================================================
# PER-PAIR VALIDATION
# =============================================================================

def validate_pair(al: pd.Series, stk: pd.Series,
                  stock_name: str,
                  returns: pd.DataFrame) -> tuple[dict, dict, dict]:
    al_ret  = returns["aluminum"]
    stk_ret = returns[stock_name]
    norm_al  = test_normality(al_ret)
    norm_stk = test_normality(stk_ret)

    # H1 — inverse beta
    h1_corr  = test_correlation(al_ret, stk_ret, norm_al, norm_stk, -1)
    h1_gran  = test_granger(returns, stock_name)
    h1_stat_al = test_stationarity(al,  "aluminum")
    h1_stat_st = test_stationarity(stk, stock_name)
    h1_spread  = build_spread_rolling(al, stk, inverse=True)
    h1_coint   = test_cointegration(-al, stk)
    h1_quality = test_spread_quality(h1_spread)
    h1_regime  = test_regime_stability(h1_spread)
    h1 = dict(name="H1 — inverse beta (cost-margin squeeze)",
              corr=h1_corr, gran=h1_gran,
              stat_al=h1_stat_al, stat_stk=h1_stat_st,
              coint=h1_coint, quality=h1_quality, regime=h1_regime,
              spread=h1_spread, norm_al=norm_al, norm_stk=norm_stk,
              score=sum([h1_corr["passes"], h1_gran["passes"],
                         h1_coint["passes"], h1_quality["passes"],
                         h1_regime["passes"]]))

    # H2 — cost-margin ratio
    ratio      = (al / stk).dropna()
    h2_quality = test_spread_quality(ratio)
    h2_regime  = test_regime_stability(ratio)
    ratio_adf_p = adfuller(ratio.dropna())[1]
    h2_stat_r   = test_stationarity(ratio, "ratio")
    h2 = dict(name="H2 — cost-margin ratio (Al / stock)",
              corr=h1_corr,
              stat_ratio=h2_stat_r,
              coint_pass=ratio_adf_p < ALPHA,
              ratio_adf_p=round(ratio_adf_p, 4),
              quality=h2_quality, regime=h2_regime, spread=ratio,
              score=sum([h1_corr["passes"],
                         ratio_adf_p < ALPHA,
                         h2_quality["passes"],
                         h2_regime["passes"],
                         not h2_stat_r["is_i1"]]))

    # H3 — positive beta (null benchmark)
    h3_corr    = test_correlation(al_ret, stk_ret, norm_al, norm_stk, +1)
    h3_spread  = build_spread_rolling(al, stk, inverse=False)
    h3_coint   = test_cointegration(al, stk)
    h3_quality = test_spread_quality(h3_spread)
    h3_regime  = test_regime_stability(h3_spread)
    h3 = dict(name="H3 — positive beta (demand-driven, null benchmark)",
              corr=h3_corr, gran=h1_gran,
              coint=h3_coint, quality=h3_quality, regime=h3_regime,
              spread=h3_spread,
              score=sum([h3_corr["passes"], h1_gran["passes"],
                         h3_coint["passes"], h3_quality["passes"],
                         h3_regime["passes"]]))
    return h1, h2, h3


# =============================================================================
# REPORT
# =============================================================================

def verdict_label(score: int) -> str:
    return ("STRONG SUPPORT"   if score >= 4 else
            "MODERATE SUPPORT" if score >= 3 else
            "WEAK SUPPORT"     if score >= 2 else
            "NOT SUPPORTED")


def format_report(stock: str, h1: dict, h2: dict, h3: dict) -> str:
    lines = []

    def row(label, val, flag=""):
        pf = f"  [{flag}]" if flag else ""
        lines.append(f"  {label:<42}{str(val):<16}{pf}")

    lines.append(f"\n{'═'*72}")
    lines.append(f"  PAIR: aluminum / {stock}")
    lines.append(f"{'═'*72}")

    for h_idx, h in enumerate([h1, h2, h3], 1):
        lines.append(f"\n  ── {h['name']} ──")

        c = h["corr"]
        lines.append(f"\n  [1] Normality → {c['method']} selected")
        lines.append(f"      Skew={h1['norm_al']['skewness'] if h_idx==1 else '—'}  "
                     f"Excess kurt={h1['norm_al']['excess_kurt'] if h_idx==1 else '—'}")
        row("Correlation r",
            f"{c['r']}  (expect {'negative' if h_idx < 3 else 'positive'})",
            "✓ direction" if c["sign_correct"] else "✗ wrong direction")
        row("p / 95% CI",
            f"{c['p']}  [{c['ci_lo']}, {c['ci_hi']}]",
            "significant" if c["significant"] else "not significant")
        row("Strength", c["strength"])

        if "gran" in h:
            g = h["gran"]
            lines.append(f"\n  [2] Granger causality (Al leads stock?)")
            row("Best lag / p-value", f"lag={g['best_lag']}  p={g['best_p']}",
                "Al LEADS ✓" if g["passes"] else "no lead ✗")

        if "stat_al" in h:
            lines.append(f"\n  [3] Stationarity")
            row("Aluminum ADF p (levels)", h["stat_al"]["adf_p"],
                "I(1) ✓" if h["stat_al"]["is_i1"] else "not I(1)")
            row(f"{stock} ADF p (levels)", h["stat_stk"]["adf_p"],
                "I(1) ✓" if h["stat_stk"]["is_i1"] else "not I(1)")

        if "coint" in h:
            ct = h["coint"]
            lines.append(f"\n  [4] Cointegration (on {'inverse ' if h_idx==1 else ''}spread)")
            row("Engle-Granger p", ct["eg_p"],
                "cointegrated ✓" if ct["eg_pass"] else "NOT ✗")
            row("Johansen trace / 5% crit",
                f"{ct['joh_trace']} / {ct['joh_crit']}",
                "cointegrated ✓" if ct["joh_pass"] else "NOT ✗")

        if "coint_pass" in h:
            lines.append(f"\n  [3] Ratio stationarity (want I(0))")
            row("Ratio ADF p", h["ratio_adf_p"],
                "stationary ✓" if h["coint_pass"] else "non-stationary ✗")

        q = h["quality"]
        lines.append(f"\n  [{'5' if 'coint' in h else '4'}] Spread/ratio quality")
        row("Half-life (days)", q["half_life_days"],
            "✓" if 5 <= q["half_life_days"] <= 60 else "too slow/fast")
        row("Hurst exponent", q["hurst_exp"],
            "mean-reverting ✓" if q["hurst_exp"] < 0.5 else "random walk ✗")
        row("Spread ADF p", q["spread_adf_p"],
            "stationary ✓" if q["stationary"] else "non-stationary ✗")
        row("Verdict", q["verdict"])

        r = h["regime"]
        lines.append(f"\n  [{'6' if 'coint' in h else '5'}] Regime stability")
        row("% windows broken", f"{r['pct_broken']}%",
            "stable ✓" if r["passes"] else "UNSTABLE ✗")
        row("Max rolling ADF p", r["max_adf_p"])

        lines.append(f"\n  {'─'*50}")
        lines.append(f"  SCORE {h['score']}/5  →  {verdict_label(h['score'])}")

    # Recommendation
    scores  = {"H1": h1["score"], "H2": h2["score"], "H3": h3["score"]}
    best_h  = max(scores, key=scores.get)
    best_s  = scores[best_h]
    actions = {
        "H1": ("Use INVERSE BETA strategy.\n"
               "  Spread = −Al − β×stock. Trade mean-reversion when |z| > 2.\n"
               "  Long stock / short Al when spread is high (Al expensive, stock cheap).\n"
               "  Reverse when spread normalises."),
        "H2": ("Use COST-MARGIN RATIO strategy.\n"
               "  Trade ratio = Al/stock. Enter long stock / short Al when ratio\n"
               "  is above its 60-day mean + 2σ (Al too expensive relative to stock).\n"
               "  Exit when ratio falls back below mean + 0.5σ."),
        "H3": ("Use POSITIVE BETA strategy (demand-driven).\n"
               "  Standard spread = Al − β×stock. Trade when |z| > 2."),
    }
    lines.append(f"\n{'═'*72}")
    lines.append(f"  RECOMMENDATION FOR {stock}")
    if best_s >= 3:
        lines.append(f"  Best: {best_h} ({verdict_label(best_s)})")
        lines.append(f"  {actions[best_h]}")
    else:
        lines.append(f"  Best score {best_s}/5 — insufficient evidence.")
        lines.append("  Do not trade this pair with real capital yet.")
        lines.append("  Try: longer history, different date range,")
        lines.append("  or different construction proxy (PKB vs XHB vs ITB).")
    lines.append(f"{'═'*72}\n")
    return "\n".join(lines)


# =============================================================================
# PLOTS
# =============================================================================

def plot_pair(fig, gs_row, stock: str,
              al: pd.Series, stk: pd.Series,
              al_ret: pd.Series, stk_ret: pd.Series,
              h1: dict, h2: dict, h3: dict) -> None:

    # Scatter
    ax0 = fig.add_subplot(gs_row[0])
    ax0.scatter(al_ret, stk_ret, alpha=0.18, s=5, color="#378ADD")
    xr = np.linspace(al_ret.min(), al_ret.max(), 100)
    m  = LinearRegression().fit(al_ret.values.reshape(-1, 1), stk_ret.values)
    col = "#E24B4A" if m.coef_[0] < 0 else "#1D9E75"
    ax0.plot(xr, m.intercept_ + m.coef_[0] * xr, color=col, lw=1.8)
    ax0.axhline(0, color="gray", lw=0.4)
    ax0.axvline(0, color="gray", lw=0.4)
    sign_txt = ("Negative β — cost squeeze" if m.coef_[0] < 0
                else "Positive β — demand-driven")
    ax0.text(0.03, 0.97, sign_txt, transform=ax0.transAxes,
             fontsize=7, va="top", color=col,
             bbox=dict(fc="white", alpha=0.6, pad=2))
    ax0.set_title(f"{stock}  β={m.coef_[0]:.3f}  ρ={h1['corr']['r']:.3f}",
                  fontsize=9)
    ax0.set_xlabel("Al return"); ax0.set_ylabel(f"{stock} return")
    ax0.grid(alpha=0.15)

    # H1 inverse spread
    ax1 = fig.add_subplot(gs_row[1])
    sp = h1["spread"].dropna()
    rm = sp.rolling(60).mean(); rs = sp.rolling(60).std()
    ax1.plot(sp.index, sp, lw=0.7, color="#534AB7", alpha=0.8)
    ax1.plot(rm.index, rm, color="gray", lw=0.9, ls="--")
    ax1.fill_between(rm.index, rm - 2 * rs, rm + 2 * rs,
                     alpha=0.10, color="#534AB7")
    ax1.set_title(f"H1 inv spread  HL={h1['quality']['half_life_days']}d  "
                  f"H={h1['quality']['hurst_exp']:.3f}", fontsize=9)
    ax1.set_ylabel("−Al − β×stock"); ax1.grid(alpha=0.15)

    # H2 ratio
    ax2 = fig.add_subplot(gs_row[2])
    rat = h2["spread"].dropna()
    rm2 = rat.rolling(60).mean(); rs2 = rat.rolling(60).std()
    ax2.plot(rat.index, rat, lw=0.7, color="#EF9F27", alpha=0.8)
    ax2.plot(rm2.index, rm2, color="gray", lw=0.9, ls="--")
    ax2.fill_between(rm2.index, rm2 - 2 * rs2, rm2 + 2 * rs2,
                     alpha=0.10, color="#EF9F27")
    ax2.set_title(f"H2 ratio  HL={h2['quality']['half_life_days']}d  "
                  f"H={h2['quality']['hurst_exp']:.3f}", fontsize=9)
    ax2.set_ylabel("Al / stock"); ax2.grid(alpha=0.15)

    # Rolling ADF p (H1 spread)
    ax3 = fig.add_subplot(gs_row[3])
    rp  = h1["regime"]["rolling_adf"]
    col_line = "#E24B4A" if h1["regime"]["pct_broken"] > 20 else "#1D9E75"
    ax3.plot(rp.index, rp, lw=0.8, color=col_line)
    ax3.axhline(0.05, color="#E24B4A", lw=1,   ls="--", label="5%")
    ax3.axhline(0.10, color="#993C1D", lw=0.7, ls=":",  label="10%")
    ax3.fill_between(rp.index, 0.10, rp.values,
                     where=rp.values > 0.10, alpha=0.2, color="#E24B4A")
    ax3.set_title(f"Rolling ADF p  ({h1['regime']['pct_broken']}% broken)",
                  fontsize=9)
    ax3.set_ylabel("p-value"); ax3.legend(fontsize=7); ax3.grid(alpha=0.15)
    ax3.set_ylim(0, min(1.0, float(rp.max()) * 1.3 + 0.05))


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def validate(al_ticker: str, stock_tickers: list[str],
             start: str = "2015-01-01") -> None:
    print(f"\nFetching: {al_ticker} vs {stock_tickers}")
    prices  = fetch(al_ticker, stock_tickers, start)
    returns = np.log(prices / prices.shift(1)).dropna()

    n   = len(stock_tickers)
    fig = plt.figure(figsize=(20, 5 * n + 2))
    gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.55, wspace=0.35)

    lines = ["=" * 72,
             "ALUMINUM / CONSTRUCTION HYPOTHESIS VALIDATION REPORT v2",
             f"Aluminum : {al_ticker}",
             f"Stocks   : {stock_tickers}",
             f"Period   : {prices.index[0].date()} → {prices.index[-1].date()}",
             "=" * 72,
             textwrap.dedent("""
CORRECTED ECONOMIC MODEL
────────────────────────
Aluminum is an INPUT COST. Rising prices squeeze builder margins.
Expected direction: NEGATIVE co-movement (not positive).

H1 — Inverse beta : spread  = −Al − β×stock
H2 — Ratio        : spread  =  Al / stock
H3 — Positive beta: spread  =  Al − β×stock  (null benchmark)
""")]

    summary = []

    for idx, stock in enumerate(stock_tickers):
        if stock not in prices.columns:
            continue
        print(f"\n  {al_ticker} / {stock} …")
        h1, h2, h3 = validate_pair(
            prices["aluminum"], prices[stock], stock, returns,
        )
        lines.append(format_report(stock, h1, h2, h3))

        best_s = max(h1["score"], h2["score"], h3["score"])
        best_h = max([("H1", h1["score"]), ("H2", h2["score"]),
                      ("H3", h3["score"])], key=lambda x: x[1])[0]
        summary.append({"Pair": stock,
                         "H1": f"{h1['score']}/5",
                         "H2": f"{h2['score']}/5",
                         "H3": f"{h3['score']}/5",
                         "Best": best_h,
                         "Verdict": verdict_label(best_s)})

        plot_pair(fig, [gs[idx, 0], gs[idx, 1], gs[idx, 2], gs[idx, 3]],
                  stock,
                  prices["aluminum"], prices[stock],
                  returns["aluminum"], returns[stock],
                  h1, h2, h3)

    lines.append("SUMMARY TABLE")
    lines.append("─" * 72)
    hdr = f"{'Pair':<8}  {'H1':>8}  {'H2':>8}  {'H3':>8}  {'Best':>5}  Verdict"
    lines.append(hdr)
    lines.append("─" * 72)
    for r in summary:
        lines.append(f"{r['Pair']:<8}  {r['H1']:>8}  {r['H2']:>8}  "
                     f"{r['H3']:>8}  {r['Best']:>5}  {r['Verdict']}")
    lines.append("─" * 72)

    report = "\n".join(lines)
    with open("hypothesis_report.txt", "w") as f:
        f.write(report)
    print("\nReport → hypothesis_report.txt")
    print(report)

    plt.suptitle(
        f"Hypothesis Validation v2 — {al_ticker} vs Construction  "
        f"[scatter | H1 inv spread | H2 ratio | rolling ADF]",
        fontsize=12, y=1.01,
    )
    plt.savefig("hypothesis_plots.png", dpi=150, bbox_inches="tight")
    print("Plots → hypothesis_plots.png")
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aluminum/Construction Hypothesis Validator v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python hypothesis_validator.py
          python hypothesis_validator.py --al AA --stocks XHB ITB PKB VMC
          python hypothesis_validator.py --start 2018-01-01
        """),
    )
    parser.add_argument("--al",     default="AA")
    parser.add_argument("--stocks", nargs="+", default=["XHB", "ITB", "PKB"])
    parser.add_argument("--start",  default="2015-01-01")
    args = parser.parse_args()
    validate(args.al, args.stocks, args.start)
