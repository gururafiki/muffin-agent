---
name: pharmaceuticals
description: >
  Common valuation principles for Healthcare - Pharmaceuticals stocks.
  Patent cliff modeling (80-90% revenue loss), pipeline probability of success
  by phase, and risk-adjusted NPV (rNPV) calculation methodology.
metadata:
  sector: pharmaceuticals
---

# Pharmaceuticals Valuation — Common Principles

## Patent Cliff Analysis (CRITICAL)

1. Identify top 5-10 drugs by revenue contribution
2. Map patent expiration dates for each
3. Model 80-90% revenue loss in 2-3 years after patent expiration
4. Assess pipeline drugs to replace lost revenue

**Example:**
- Blockbuster drug: $3B annual revenue, patent expires Year 3
- Expected Year 4 revenue: $300-600M (83-85% loss)
- Company must have pipeline drugs at advanced stage to offset

## Pipeline Probability of Success (PoS)

| Phase | Cumulative PoS to Approval |
|-------|---------------------------|
| Phase 1 | ~30% |
| Phase 2 | ~30% |
| Phase 3 | ~60% |
| Regulatory (NDA/MAA) | ~90% |

## Risk-Adjusted NPV (rNPV) Formula

```
rNPV = Σ [Commercial CF_t × Cumulative P(success to year t)] / (1+r)^t
```

- CF = expected commercial cash flows
- P = probability-weighted by phase
- Discount rate 8-12% typical (lower than standard WACC; risk captured in probabilities)

## Drug Portfolio Analysis

- Analyze major drugs individually, not as aggregate
- Patent expiration timeline: map revenue cliff by drug
- New drug peak sales: Market size × expected penetration rate
- Realistic peak: $2-3B for successful differentiated drug (discount for side effects, competition, pricing pressure)
