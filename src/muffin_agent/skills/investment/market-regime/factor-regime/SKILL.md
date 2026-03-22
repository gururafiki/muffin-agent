---
name: factor-regime
description: >
  Classify Fama-French factor regimes using Z-score thresholds and determine
  factor tilts (value, quality, momentum, size). Use when you have factor
  return data (HML, SMB, MOM, RMW, CMA trailing returns and historical
  statistics) and need to assess which style factors are favoured.
---

# Factor Regime Classification

## Z-Score Computation

Call `compute_factor_zscore` once per factor with:
- `factor_name`: e.g. "HML", "SMB", "MOM", "RMW", "CMA"
- `trailing_12m`: trailing 12-month cumulative return
- `mean_60m`: 5-year mean of 12-month rolling returns
- `std_60m`: 5-year standard deviation of 12-month rolling returns

All 5 calls can be issued in parallel.

## Z-Score Thresholds

| |Z| Range | Signal | Interpretation |
|---|---|---|---|
| > 2.0 | **Extreme** | Factor in a strong regime; high-conviction tilt warranted |
| 1.5–2.0 | **Significant** | Notable deviation; directional tilt supported |
| 0.5–1.5 | **Moderate** | Some signal but not decisive; neutral-to-mild tilt |
| < 0.5 | **Weak** | Factor near historical mean; no directional call |

**Sign interpretation**: Positive Z = factor outperforming (tailwind for that style);
negative Z = factor underperforming (headwind).

## Factor Tilt Logic

### Value (HML)
- **Tailwind** when: Z > +1.0 AND growth_cycle is `slowing` or `recovering`
  (value rotation typically occurs in reflationary regimes)
- **Headwind** when: Z < -1.0 OR growth_cycle is `contracting` with risk_off
  (value traps in recession)
- **Neutral**: otherwise

### Quality / Profitability (RMW)
- **Tailwind** when: Z > +0.5 OR liquidity is `risk_off` / `crisis`
  (quality is defensive; outperforms in stress)
- **Headwind** when: Z < -1.0 AND liquidity is `risk_on`
  (cyclicals rip, quality lags)
- **Neutral**: otherwise

### Momentum (MOM)
- **Tailwind** when: Z > +1.0 AND VIX regime is `normal` or `complacency`
  (trending markets favour momentum)
- **Headwind** when: Z < -1.0 OR VIX regime is `crisis`
  (reversals and regime breaks kill momentum)
- **Neutral**: otherwise

### Size (SMB)
- **Tailwind** when: Z > +1.0 AND growth_cycle is `expanding` or `recovering`
  AND credit spreads < 40th percentile
  (small caps outperform early-cycle with easy credit)
- **Headwind** when: Z < -0.5 OR credit spreads > 80th percentile
  (small caps are first to suffer in credit stress)
- **Neutral**: otherwise

## Cross-Factor Consistency Checks

After assigning tilts, verify:
1. Momentum tailwind + VIX crisis is unusual — justify or revise
2. Value tailwind + quality headwind is common in risk-on rotations — acceptable
3. Size tailwind + credit stress is contradictory — the credit signal dominates
4. All four factors at headwind suggests broad risk-off — confirm with liquidity dimension
