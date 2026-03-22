---
name: yield-curve-analysis
description: >
  Interpret yield curve shape, slope metrics, credit spreads, and policy rate
  signals. Use when you have fixed-income data (Treasury yields, TIPS
  breakevens, IG/HY OAS, EFFR) and need to classify monetary policy stance
  and credit conditions.
---

# Yield Curve Analysis

## Yield Curve & Policy Rate Computation

Call `compute_yield_curve_metrics` with yield_10y, yield_2y, yield_3m,
tips_breakeven_10y, and effr. Returns JSON with:
- `slope_10y2y_bps` — 10Y minus 2Y in basis points (negative = inverted)
- `slope_10y3m_bps` — 10Y minus 3M in basis points
- `real_yield_10y_bps` — 10Y yield minus TIPS breakeven
- `policy_rate_distance_bps` — 10Y minus EFFR

## Shape Classification

| slope_10y2y_bps | slope_10y3m_bps | Shape | Interpretation |
|---|---|---|---|
| > +50 | > +100 | **normal** | Economy healthy, term premium positive |
| +10 to +50 | +10 to +100 | **flat** | Late-cycle or transition; ambiguous signal |
| -10 to +10 | any | **flat** | Uncertainty; often precedes inversion |
| < -10 | < 0 | **inverted** | Recession signal; market expects rate cuts |
| < 0 | > +50 | **humped** | Mid-curve stress; front-end pricing near-term cuts |

## Trend Classification

Compare current slope to 3-month-ago slope:
- **steepening**: slope widened by > +15 bps — bullish for risk assets
- **flattening**: slope narrowed by > 15 bps — caution signal
- **stable**: change within ±15 bps

## Credit Spread Interpretation

Calculate IG and HY OAS percentiles against 5-year history using sandbox:

```python
from scipy.stats import percentileofscore
ig_oas_pctile = percentileofscore(ig_oas_5y_history, ig_oas_current)
hy_oas_pctile = percentileofscore(hy_oas_5y_history, hy_oas_current)
```

| Percentile | Signal | Implication |
|---|---|---|
| > 80th | Elevated stress | Risk-off; credit conditions tightening |
| 40th–80th | Normal | Neutral credit conditions |
| < 20th | Complacency | Risk-on; spreads near cycle lows |

IG-HY spread divergence (HY widening faster than IG) signals distress
migrating down the quality spectrum — especially concerning in late-cycle.

## Real Yield Signal

| real_yield_10y_bps | Interpretation |
|---|---|
| > +200 | Restrictive — headwind for growth/duration assets |
| +50 to +200 | Moderately tight — watch for growth impact |
| 0 to +50 | Neutral |
| < 0 | Accommodative — supports risk assets |

## Policy Rate Distance

`policy_rate_distance_bps` = 10Y yield minus EFFR:
- **Positive (> +50)**: Market pricing expansion/inflation above current policy
- **Near zero (±50)**: Curve pricing policy as appropriate
- **Negative (< -50)**: Market pricing rate cuts ahead — recession signal
