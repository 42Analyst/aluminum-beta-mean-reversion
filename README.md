# Aluminum / Construction Statistical Arbitrage Research

**Project status**: Hypothesis rejected after testing (April 2026)

Methodology and implementation are original work. Reuse with attribution appreciated :)

### Project Goal
Test whether aluminum prices (as an input cost) have a exploitable statistical relationship with US construction/homebuilder stocks for a mean-reversion / pairs trading strategy.

**Initial flawed hypothesis**: Positive correlation — aluminum and construction stocks move together due to demand (more building = more aluminum needed).

**Corrected hypothesis** (after realizing the economic reality):
- Aluminum is primarily an **input cost** (siding, wiring, frames, HVAC, etc.).
- Rising aluminum prices should **squeeze builder margins** → negative relationship with stock prices.

I formalized this into three competing hypotheses:

- **H1 — Inverse Beta (Cost Squeeze)**: spread = −Al − β × stock  
- **H2 — Cost-Margin Ratio**: ratio = Al price / stock price  
- **H3 — Positive Beta** (original demand-driven idea, kept as null benchmark)

### Methodology
I built a comprehensive hypothesis validation framework (`aluminum hypothesis.py`) that tests:

- Directional correlation (Spearman/Pearson with bootstrap CI)
- Granger causality (does aluminum lead stocks?)
- Stationarity (ADF + KPSS)
- Cointegration (Engle-Granger + Johansen)
- Spread quality (half-life, Hurst exponent, ADF on spread)
- Regime stability (rolling ADF windows)

All tests use rolling hedge ratios to avoid look-ahead bias. Full technical documentation is in `StatArb_Methodology.docx`.

### Key Results (Period: 2015–2026)

**Summary Table**

| Pair | H1 (Inverse) | H2 (Ratio) | H3 (Positive) | Best | Verdict      |
|------|--------------|------------|---------------|------|--------------|
| XHB  | 1/5          | 0/5        | 2/5           | H3   | Weak Support |
| ITB  | 1/5          | 0/5        | 2/5           | H3   | Weak Support |
| PKB  | 1/5          | 0/5        | 2/5           | H3   | Weak Support |

**Main findings**:
- Correlation exists but is **positive** (moderate, ~0.34–0.45), not the expected negative for a cost-squeeze story.
- No meaningful cointegration on the inverse spread or ratio.
- Spreads are **not tradeable** (long half-lives 200–500+ days, Hurst ≈ 1.03 → near random walk, high regime instability ~85–92% broken windows).
- Granger causality shows aluminum somewhat leads stocks at lag 4, but not enough to build a reliable edge.
- H3 (positive beta) performed best — suggesting demand-driven effects sometimes dominate, but still insufficient for a stat-arb strategy.

**Conclusion**: The statistical foundation for this pairs trade does **not hold** with daily data on these instruments. Do not trade this pair with real capital.

### Results

All three hypotheses (inverse beta, cost-margin ratio, 
demand-driven) scored ≤ 2/5 on the validation framework.

## Why it failed (and what that tells us)

The core issue is signal dilution: aluminum represents ~3–7% of residential construction material cost. At that weight, a 30%+ aluminum spike produces only 1–2pp of margin compression  — far below the noise floor of daily equity returns driven by rate decisions, housing starts, and broad market beta.

### What I'd test next
- Monthly frequency (mechanism is quarterly, not daily)
- FRED aluminum PPI vs NAHB sentiment index
- Copper or lumber as higher-weight construction inputs

### Repository Contents
- `aluminum hypothesis.py` — Standalone hypothesis testing engine (run this first)
- `hypothesis_report.txt` — Full detailed output from latest run
- `hypothesis_plots.png` — Visual diagnostics (scatter, spreads, rolling ADF)
- `StatArb_Methodology.docx` — Complete technical documentation & pipeline design

### How to Run
```bash
python hypothesis_validator.py --al AA --stocks XHB ITB PKB
# or with aluminum futures (requires proper roll adjustment)
# python hypothesis_validator.py --al ALI=F --stocks XHB ITB PKB
