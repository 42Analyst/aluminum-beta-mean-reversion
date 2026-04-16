## Aluminum Beta Mean-Reversion Strategy

## Overview
This project explores a statistical arbitrage idea based on the historical relationship between aluminum futures prices and construction-sector stocks. 

The core hypothesis: Aluminum prices exhibit mean-reverting behavior relative to a basket of construction companies (the "aluminum beta"). When the spread deviates significantly from its historical average, a potential trading signal is generated.

## Why This Matters
Aluminum is heavily tied to construction, infrastructure, autos, and aerospace. In 2026, tariffs and reflation have pushed aluminum prices to multi-year highs, creating more frequent deviations worth testing. This is a classic commodities-to-equities pairs/mean-reversion approach.

## Hypothesis to Test
- H0: Aluminum prices and construction stocks maintain a stable long-term correlation (aluminum beta).
- Test for statistical significance of mean reversion in the spread.
- Define clear entry/exit rules based on z-score or deviation thresholds.

## Data Sources (planned)
- Aluminum futures: CME HG or LME data
- Construction stocks: ETF or basket (e.g., ITB, XHB, or individual names like Caterpillar, Vulcan Materials, etc.)
- Timeframe: Daily or weekly data (5–10 years)

## Next Steps
1. Pull historical price data
2. Calculate aluminum beta and spread
3. Run hypothesis tests (ADF test for stationarity, correlation analysis)
4. Backtest simple trading rules
5. Visualize results and document limitations
