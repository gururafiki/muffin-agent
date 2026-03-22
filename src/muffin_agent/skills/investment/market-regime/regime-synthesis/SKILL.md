---
name: regime-synthesis
description: >
  Score and classify the 4 macro regime dimensions (growth, inflation,
  monetary policy, liquidity) and synthesise into a regime label with
  positioning guidance. Use after collecting macro data and computing
  yield curve metrics, factor Z-scores, VIX regime, and credit spread
  percentiles.
---

# Regime Synthesis

## Dimension Scoring Rubrics

For each dimension, follow Chain-of-Thought reasoning: list the relevant data
points, state the scale anchors, then assign the label and score.

### Dimension 1: Growth / Activity Cycle

**Score scale**: 1.0 = strong expansion → 0.5 = neutral/stalling → 0.0 = deep contraction

| Score Range | Label | Anchors |
|---|---|---|
| 0.8–1.0 | `expanding` | GDP > 3% annualised, PMI > 55, CLI rising, payrolls strong |
| 0.5–0.8 | `slowing` | GDP 1–3%, PMI 50–55 declining, CLI flat, payrolls decelerating |
| 0.2–0.5 | `recovering` | GDP turning positive, PMI crossing 50 upward, CLI inflecting |
| 0.0–0.2 | `contracting` | GDP < 0%, PMI < 45, CLI falling sharply, unemployment rising |

Key indicators: real GDP QoQ annualised, CLI trend, PMI level, unemployment
rate direction, payrolls trend, `slope_10y3m_bps` (inverted = recession signal),
`copper_mom_pct` (growth proxy).

### Dimension 2: Inflation / Price Regime

**Score scale**: 1.0 = severe inflation → 0.5 = moderate/on-target → 0.0 = deflation

| Score Range | Label | Anchors |
|---|---|---|
| 0.8–1.0 | `high_rising` | CPI > 6% YoY, accelerating, breakevens rising |
| 0.6–0.8 | `elevated_stable` | CPI 3–6% YoY, stable or decelerating slowly |
| 0.4–0.6 | `moderate` | CPI 2–3% YoY, near target, breakevens anchored |
| 0.2–0.4 | `low_falling` | CPI 0–2% YoY, decelerating, breakevens falling |
| 0.0–0.2 | `deflationary` | CPI < 0%, commodity prices falling, demand collapse |

Key indicators: headline CPI YoY, core CPI YoY, `cpi_3m_annualised_pct`,
`cpi_reacceleration`, PCE trend, TIPS breakevens, oil/commodity prices.

### Dimension 3: Monetary Policy Stance

**Score scale**: 1.0 = aggressively tightening → 0.5 = neutral → 0.0 = aggressively easing

| Score Range | Label | Anchors |
|---|---|---|
| 0.8–1.0 | `aggressively_tightening` | Hiking > 100bps/year, QT in progress, hawkish tone |
| 0.6–0.8 | `tightening` | Hiking < 100bps, or paused at restrictive level |
| 0.4–0.6 | `neutral` | On hold at neutral rate, balanced FOMC tone |
| 0.2–0.4 | `easing` | Cutting < 100bps, or paused with dovish guidance |
| 0.0–0.2 | `aggressively_easing` | Cutting > 100bps, QE active, emergency measures |

Key indicators: EFFR level and direction, FOMC tone, `slope_10y2y_bps`
(inverted = past tightening absorbed), `real_yield_10y_bps`,
`policy_rate_distance_bps`, balance sheet policy.

### Dimension 4: Liquidity / Risk Appetite

**Score scale**: 1.0 = extreme risk-on → 0.5 = neutral → 0.0 = crisis/risk-off

| Score Range | Label | Anchors |
|---|---|---|
| 0.8–1.0 | `risk_on` | IG OAS < 20th pctile, VIX < 15, equity rally, dollar weak |
| 0.6–0.8 | `cautiously_risk_on` | Spreads below median, VIX normal, selective risk-taking |
| 0.4–0.6 | `neutral` | Mixed signals, VIX 15–20, spreads near median |
| 0.2–0.4 | `risk_off` | VIX > 25, IG OAS > 60th pctile, gold rallying, dollar strong |
| 0.0–0.2 | `crisis` | VIX > 35, HY OAS > 80th pctile, flight to safety, dollar surge |

Key indicators: VIX level and `vix_regime`, `ig_oas_pctile`, `hy_oas_pctile`,
`gold_mom_pct`, `usd_mom_pct`, equity index multiple, Mkt-RF factor return.

## Regime Label Construction

After scoring all 4 dimensions, synthesise into a 3–6 word plain-English label
combining the most salient dimensions. Examples:

| Growth | Inflation | Policy | Liquidity | Label |
|---|---|---|---|---|
| expanding | moderate | neutral | risk_on | "Goldilocks expansion" |
| slowing | elevated_stable | tightening | risk_off | "Stagflationary slowdown" |
| contracting | low_falling | easing | crisis | "Recession / Fed pivot" |
| recovering | moderate | easing | cautiously_risk_on | "Early-cycle reflation" |
| expanding | high_rising | aggressively_tightening | neutral | "Overheating, hawkish Fed" |

## Positioning Guidance

Translate the regime into concrete, regime-calibrated guidance:

### Beta Range
| Regime Character | Beta Range |
|---|---|
| Risk-on expansion | 0.9–1.2 |
| Neutral / mixed | 0.8–1.0 |
| Late-cycle / uncertain | 0.7–0.9 |
| Risk-off / contraction | 0.5–0.8 |
| Crisis | 0.3–0.6 |

### Exposure Caps
If regime is **adverse** (liquidity = `risk_off` / `crisis`, OR growth =
`contracting`), explicitly state:

> "Adverse regime — require idiosyncratic alpha justification before adding
> risk. Recommended net exposure cap: [X]%, beta cap: [Y]."

### Sector & Style Tilts
- **Sector tilts**: 1–2 favoured and 1–2 avoided sectors grounded in the regime
- **Style tilts**: e.g., "quality over growth; value vs. growth neutral"
- Align with factor tilt conclusions from factor-regime analysis
