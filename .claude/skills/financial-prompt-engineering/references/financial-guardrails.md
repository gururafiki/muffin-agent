# Financial Guardrails

Detailed patterns for preventing common failures in financial LLM prompts: hallucination, calculation errors, look-ahead bias, poor confidence calibration, and data degradation.

---

## 1. Hallucination Prevention

### The Problem
LLMs fabricate financial data with high confidence and fluent presentation. In finance, a single hallucinated number can cascade through calculations and produce completely wrong conclusions.

### Types of Financial Hallucination
- **Intrinsic:** Output contradicts data in the source (e.g., says revenue grew when it declined)
- **Extrinsic:** Introduces facts not present in any source (e.g., invents an EPS number)
- **Omission:** Fails to mention rapidly-changing information (earnings revisions, regulatory changes)

### Prevention Architecture

**Layer 1: Source Grounding**
Add to every analysis prompt:
```
Every quantitative claim must cite its data source:
- What tool or filing provided the data
- What time period it covers
- The exact number used

Example: "Revenue of $94.9B (equity_fundamentals_income, FY2023 10-K)"
```

**Layer 2: No-Fabrication Clause**
Include verbatim in every prompt:
```
If data is unavailable from any tool, state it is unavailable.
NEVER estimate, approximate, or fabricate numbers.
Do NOT fill data gaps with plausible-sounding values.
```

**Layer 3: Structured Verification**
Add as a workflow step:
```
Verification step:
1. For each number in your analysis, confirm it came from a tool response
2. For each calculation, verify the inputs match the tool outputs
3. Flag any number that cannot be traced to a specific data source
```

**Combined effect:** CoT + structured verification reduces hallucination by ~75% vs standard prompting.

---

## 2. Calculation Error Prevention

### The Problem
LLMs make arithmetic errors, mix units, and apply formulas incorrectly. These errors compound in multi-step financial calculations.

### Formula-First Pattern
Require the agent to declare the formula before calculating:
```
When calculating any financial metric:
1. State the formula: "Free Cash Flow = Operating Cash Flow - Capital Expenditures"
2. Identify the inputs: "OCF = $110.5B, CapEx = $11.0B"
3. Calculate: "FCF = $110.5B - $11.0B = $99.5B"
4. Sanity check: "FCF of $99.5B on $383B revenue = 26% FCF margin, reasonable for tech"
```

### Sanity Check Ranges
Include plausible ranges for common metrics to catch obvious errors:

| Metric | Plausible Range | Red Flag If |
|--------|----------------|-------------|
| P/E Ratio | 5x - 100x | Negative (loss-making needs different treatment) or >200x |
| EV/EBITDA | 3x - 50x | Negative EBITDA or >75x |
| Revenue Growth (YoY) | -50% to +200% | >500% (likely wrong period or units) |
| Operating Margin | -100% to +80% | >90% (possible data error) |
| Debt/Equity | 0x - 10x | >20x (possible units mismatch) |
| ROE | -100% to +100% | >150% (possible negative equity distortion) |
| Dividend Yield | 0% - 15% | >20% (likely price crash or data error) |
| Beta | -1.0 to 3.0 | >5.0 (extreme, verify) |
| FCF Margin | -50% to +50% | >60% (verify — very few companies achieve this) |
| Current Ratio | 0.2 - 10.0 | >20 (likely error) |

**Prompt instruction:**
```
After calculating any metric, verify it falls within plausible ranges.
If a calculated value seems extreme, re-check inputs and formula.
If confirmed correct, explicitly note it as unusual and explain why.
```

### Unit Consistency
```
Before any calculation involving multiple data points:
- Verify all amounts use the same unit (millions vs billions)
- Verify all amounts use the same currency
- Verify time periods match (don't mix quarterly and annual figures)
- State units explicitly in your output
```

### Deterministic Computation
For complex multi-step calculations (DCF, WACC, sensitivity analysis):
```
For calculations with more than 3 steps, write Python code to compute the result.
Show the code, run it, and report the output. This eliminates arithmetic errors.
```

---

## 3. Look-Ahead Bias Prevention

### The Problem
LLMs are trained on data from the future relative to any historical analysis date. This means they "know" earnings results, price movements, and events that hadn't happened yet at the analysis date. 54% of finance professionals rate this as "extremely critical."

### Temporal Checkpoint Pattern
Add to any analysis prompt that involves a specific date:

```
TEMPORAL RULES — Analysis date: {DATE}

1. BEFORE ANALYZING: List what data sources were available as of {DATE}:
   - Most recent 10-K filed: {determine from filing dates}
   - Most recent 10-Q filed: {determine from filing dates}
   - Latest earnings call: {determine from dates}
   - Available analyst estimates: {published before DATE only}

2. DURING ANALYSIS: For every data point used, verify:
   - Was this data publicly available before {DATE}?
   - If this is a forecast/estimate, was it published before {DATE}?

3. PROHIBITED: Do NOT reference:
   - Earnings results reported after {DATE}
   - Price movements after {DATE}
   - News events after {DATE}
   - Analyst revisions published after {DATE}

4. AFTER ANALYSIS: In your reflection step, explicitly confirm:
   "All data points used predate the analysis date of {DATE}."
```

### Point-in-Time Anchoring
Even for current-date analysis, anchor explicitly:
```
Analyze as of today's date. Use only currently available data.
Do not reference future events, expected earnings, or forward guidance
as if they are known facts.
```

---

## 4. Confidence Calibration

### The Problem
LLM confidence scores are poorly calibrated. Expected Calibration Error ranges from 0.12-0.40 across models. This means a model saying it's "80% confident" is often wrong 30-50% of the time.

### Multi-Dimensional Confidence Breakdown
Instead of a single confidence number, break down:

```
Confidence Assessment:
- Data Sufficiency: {High/Medium/Low}
  {What data was obtained vs what was needed}
- Data Recency: {High/Medium/Low}
  {How current is the most recent data point}
- Source Quality: {High/Medium/Low}
  {Official filings vs estimates vs news vs social}
- Coverage: {High/Medium/Low}
  {Were all dimensions adequately covered?}

Overall Confidence: {weighted assessment}
```

### Calibration Guidelines
Provide concrete examples of what each level means:

```
HIGH confidence requires:
- All primary data points obtained from official filings
- Data less than 90 days old
- No critical gaps in coverage
- Multiple data sources confirm key findings

MEDIUM confidence when:
- Some data points from estimates/third-party sources
- Data 90-180 days old
- Minor gaps in coverage (non-critical dimensions missing)
- OR one dimension has conflicting signals

LOW confidence when:
- Multiple data points unavailable or from low-quality sources
- Data more than 180 days old
- Critical gaps in coverage
- Major conflicting signals across dimensions
- OR analysis relies heavily on assumptions
```

### Epistemic Uncertainty Flag
```
If you are uncertain about a conclusion, explicitly flag it:
"UNCERTAIN: {claim}. This is based on {limited data / assumption / extrapolation}
and may change materially with {what additional data would resolve it}."
```

---

## 5. Progressive Data Degradation

### The Problem
Financial data is often incomplete — tools fail, providers lack coverage, data is delayed. Agents must handle this without fabricating or silently skipping.

### Graceful Degradation Pattern
```
When data is partially unavailable:

1. ACKNOWLEDGE: State specifically what data is missing and from which tool.
   "FMP provider unavailable for AAPL balance sheet. equity_fundamentals_balance
   returned no data."

2. PROCEED: Continue with available data.
   "Proceeding with income statement and cash flow data. Balance sheet metrics
   (Debt/Equity, Current Ratio) will be excluded from this analysis."

3. ADJUST: Modify scoring weights to reflect what's available.
   "Redistributing Balance Sheet weight (0.25) proportionally:
   Profitability: 0.30 → 0.40, Cash Generation: 0.25 → 0.33, Growth: 0.20 → 0.27"

4. MARK: Note the limitation in output.
   "Analysis limited by: Missing balance sheet data. Leverage and liquidity
   assessment not possible. Overall confidence reduced to Medium."

5. NEVER: Do not estimate missing data. Do not silently skip it.
```

### Tool Failure Escalation
```
If tool fails:
1. Note briefly: "{tool_name} unavailable"
2. Try different parameters or alternative tool (NEVER same args)
3. If no alternative, mark data as unavailable
4. Continue with remaining tools
5. In summary, list what could not be retrieved and how it affects the analysis
```

---

## 6. Common Anti-Patterns

### 1. Vague Data Requests
- **Bad:** "Get financial data for AAPL"
- **Good:** "Get AAPL FY2023 10-K: Revenue, Net Income, Operating Margin, FCF from consolidated financial statements"
- **Fix:** Specify exact metrics, filing type, and fiscal period

### 2. Unbounded Time Ranges
- **Bad:** "Get historical prices"
- **Good:** "Get daily closing prices from 2023-01-01 to 2024-03-31"
- **Fix:** Always include start_date and end_date

### 3. Mixing Time Periods
- **Bad:** Comparing Q3 2023 to Q4 2022
- **Good:** Comparing Q3 2023 to Q3 2022 (YoY) or Q2 2023 to Q3 2023 (sequential)
- **Fix:** Specify comparison type explicitly (YoY or sequential)

### 4. No Source Attribution
- **Bad:** "Revenue grew 15%"
- **Good:** "Revenue grew 15% YoY to $94.9B (Q3 2024 10-Q, filed 2024-11-01)"
- **Fix:** Require source, period, and filing date for every claim

### 5. Overfitting to Single Data Point
- **Bad:** "Stock down 5% today, therefore bearish"
- **Good:** "Short-term decline (-5%) in context of +25% 6-month return and stable fundamentals. Assess whether decline is noise or signal."
- **Fix:** Require multi-factor analysis, not single-point conclusions

### 6. Ignoring Data Quality
- **Bad:** Using analyst estimates from 6 months ago as if current
- **Good:** "Consensus estimates last updated 2024-02-15. Recent earnings (2024-03-01) not yet reflected in estimates."
- **Fix:** Include recency check in confidence calibration

### 7. Narrative-Only Scoring
- **Bad:** "The company looks strong, score: 0.8"
- **Good:** "ROE 24.3% vs sector median 15.1%, margin expansion +2pp YoY, FCF conversion 1.15x -> Quality score: 0.78"
- **Fix:** Require formula → data → calculation → score chain

### 8. Silent Data Gaps
- **Bad:** (Agent skips balance sheet analysis without mentioning it)
- **Good:** "Balance sheet data unavailable from provider. Leverage and liquidity assessment excluded. Confidence reduced."
- **Fix:** Require explicit gap acknowledgment and confidence adjustment

---

## Quick Checklist

Copy this into any PR review for prompt changes:

```
Prompt Review Checklist:
[ ] Role declaration is specific and unambiguous
[ ] Tool/subagent descriptions include "Use for..." guidance
[ ] Workflow steps are numbered and explicit
[ ] Source grounding required for quantitative claims
[ ] No-fabrication clause included
[ ] Formula-first calculation pattern specified
[ ] Sanity ranges for relevant metrics included
[ ] Temporal anchoring present (if applicable)
[ ] Confidence breakdown is multi-dimensional
[ ] Data degradation handled (acknowledge, proceed, mark, adjust)
[ ] Error handling block present (data collection) or reflection step (analysis)
[ ] Output format structured and defined
[ ] Token count within optimal range (150-300 data collection, 400-600 analysis, 500-700 orchestrators)
```