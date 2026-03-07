# Prompting Techniques for Financial Agents

Detailed patterns with financial examples for each prompting technique. Choose based on the agent's task type and complexity.

## ReAct (Reasoning + Acting)

**What it is:** Interleaved Thought → Action → Observation loop. The model reasons about what to do, takes an action (tool call), observes the result, then reasons about what to do next.

**Best for:** Data collection agents that call tools.

**When to use:** Any agent whose primary job is retrieving data via MCP tools. Already built into `create_agent` from `langchain.agents`.

**Financial example:**
```
Thought: I need to get AAPL's current valuation multiples and compare to peers.
         I'll start with the current quote for price data.
Action: equity_price_quote(symbol="AAPL")
Observation: Price: $178.50, Market Cap: $2.78T, P/E: 29.3, ...
Thought: Now I need peer multiples. Let me get fundamental ratios for MSFT and GOOGL.
Action: equity_fundamentals_ratios(symbol="MSFT")
Observation: P/E: 35.2, EV/EBITDA: 24.1, ...
```

**Prompt pattern:**
```
When given a request:
1. Identify what data is needed and which tools can provide it.
2. Call tools with specific parameters.
3. If a tool fails, try alternative tools or parameters (never retry same args).
4. Summarize findings after all calls complete.
```

**Key best practices:**
- Keep the Thought step focused on tool selection, not analysis
- Include termination condition: "After all relevant data is collected, summarize"
- Don't overload the initial prompt — let the loop handle complexity
- Provide clear tool descriptions so the agent picks the right one

---

## Chain-of-Thought (CoT)

**What it is:** Explicit step-by-step reasoning before reaching a conclusion. The model shows its work at each step.

**Best for:** Linear analysis tasks with sequential logic.

**When to use:** Earnings analysis, ratio calculation, compliance checking, single-criterion evaluation.

**Accuracy gain:** +23-45% on financial forecasting tasks vs direct prompting (AI in Investment Analysis, 2024).

**Financial example — Earnings Quality Analysis:**
```
Step 1: Identify relevant metrics for earnings quality assessment.
  Required: Net Income, Operating Cash Flow, Accruals Ratio, Revenue Recognition patterns.

Step 2: Collect data.
  Net Income (FY2023): $94.7B
  Operating Cash Flow (FY2023): $110.5B
  Total Assets (FY2023): $352.6B, (FY2022): $338.5B

Step 3: Calculate earnings quality metrics.
  Cash Conversion Ratio = OCF / Net Income = $110.5B / $94.7B = 1.167
  Interpretation: >1.0 indicates high-quality earnings (cash exceeds reported income)

  Accruals Ratio = (Net Income - OCF) / Average Total Assets
  = ($94.7B - $110.5B) / (($352.6B + $338.5B) / 2)
  = -$15.8B / $345.55B = -0.046
  Interpretation: Negative accruals = earnings driven by cash, not accounting

Step 4: Compare to benchmarks.
  S&P 500 median Cash Conversion: 0.95
  Company: 1.167 (top quartile)

Step 5: Score.
  Earnings Quality Score: 0.82 / 1.0
  Reasoning: Cash conversion well above median, negative accruals confirm
  earnings quality. Score reflects strong cash generation exceeding reported profits.
```

**Prompt pattern:**
```
Analyze {criterion} using the following steps:
1. Define what metrics are needed
2. Collect/identify the relevant data points
3. Calculate metrics (state the formula, then calculate, then interpret)
4. Compare to benchmarks (peer median, historical average, sector norm)
5. Score 0.0-1.0 with explicit reasoning tied to data
```

**Key best practices:**
- Always require the formula before the calculation
- Each step should produce a concrete output (a number, a comparison, a conclusion)
- Include interpretation after each calculation (what does this number mean?)
- Don't say "think step by step" — provide the actual steps

---

## Tree-of-Thought (ToT)

**What it is:** Generate multiple reasoning paths (branches), evaluate each, prune less promising ones, and select the best.

**Best for:** Tasks where multiple valid approaches exist and trade-offs must be evaluated.

**When to use:** Comparing valuation methodologies, scenario analysis, portfolio optimization with constraints.

**Financial example — Valuation Method Selection:**
```
Question: What is the intrinsic value of AAPL?

Branch A: DCF Valuation
  Assumptions: FCF growth 8%, WACC 9.5%, terminal growth 3%
  Result: $195 per share
  Confidence: Medium (sensitive to growth/WACC assumptions)

Branch B: Relative Valuation (P/E)
  AAPL forward P/E: 28x, Peer median: 25x
  Applying peer median to AAPL EPS: 25 × $6.80 = $170
  Result: $170 per share
  Confidence: Medium (peer selection matters)

Branch C: Relative Valuation (EV/EBITDA)
  AAPL EV/EBITDA: 22x, Peer median: 20x
  Implied EV at peer multiple: $380B, minus net debt...
  Result: $165 per share
  Confidence: Medium-High (less affected by capital structure)

Evaluation: Methods converge in $165-195 range.
  DCF gives upside case, multiples give base case.
  Weighted estimate: $178 (DCF 40%, P/E 30%, EV/EBITDA 30%)
  Current price: $178.50 → Fair value assessment: fairly valued
```

**Prompt pattern:**
```
Evaluate {question} using multiple approaches:
1. Generate 2-3 independent analytical approaches
2. For each approach:
   a. State assumptions explicitly
   b. Show calculation steps
   c. Produce a result
   d. Rate confidence in this approach
3. Compare results across approaches
4. Identify convergence/divergence
5. Produce weighted final assessment with justification for weights
```

---

## Graph-of-Thought (GoT)

**What it is:** Multiple reasoning paths that can merge, aggregate, and build on each other (not just independent branches). Creates a graph structure where insights from one path inform others.

**Best for:** Multi-dimensional stock analysis where factors interact.

**Accuracy gain:** +15-25% accuracy, -25-30% hallucination rate vs baseline (Joshi, 2025).

**When to use:** Multi-factor stock scoring, criteria synthesis, any analysis combining technical + fundamental + sentiment + macro factors.

**Financial example — Multi-Dimensional Stock Assessment:**
```
Path A: Fundamental Quality
  Data: ROE 24%, margin expansion +2pp YoY, FCF conversion 1.15x
  → Quality Score: 0.78

Path B: Growth Assessment
  Data: Revenue CAGR 8%, EPS growth 12%, TAM expansion into services
  → Growth Score: 0.72

Path C: Valuation
  Data: Forward P/E 28x vs peer median 25x, PEG 2.3x
  → Valuation Score: 0.45 (premium to peers)

Path D: Risk Assessment
  Data: Net cash position, AA+ credit, beta 1.2, minimal regulatory risk
  → Risk Score: 0.85 (low risk)

Aggregation (paths merge):
  Fundamental quality (0.78) SUPPORTS the valuation premium (Path A informs Path C)
  → Adjusted Valuation: 0.55 (premium partially justified by quality)

  Strong balance sheet (Path D) REDUCES growth risk (Path D informs Path B)
  → Adjusted Growth: 0.75 (growth has financial backing)

Synthesis:
  Overall = Quality(0.30) × 0.78 + Growth(0.25) × 0.75 + Valuation(0.25) × 0.55 + Risk(0.20) × 0.85
  = 0.234 + 0.188 + 0.138 + 0.170 = 0.73 (Moderately Bullish)
```

**Prompt pattern:**
```
Analyze {subject} across multiple dimensions that inform each other:

1. Score each dimension independently:
   - {Dimension A}: Score based on {criteria}
   - {Dimension B}: Score based on {criteria}
   - {Dimension C}: Score based on {criteria}
   - {Dimension D}: Score based on {criteria}

2. Identify cross-dimensional interactions:
   - How does {Dim A} affect the interpretation of {Dim C}?
   - How does {Dim D} modify the risk profile of {Dim B}?
   - Adjust scores where interactions justify it, explaining why

3. Synthesize with explicit weights:
   Overall = Sum of (weight × adjusted_score) for each dimension
   Show the arithmetic. Weights must sum to 1.0.
```

---

## Self-Consistency

**What it is:** Generate multiple independent analyses of the same question, then identify the consensus view. Different from ToT because it uses the same approach multiple times (not different approaches).

**Best for:** High-stakes recommendations where you want to reduce variance.

**When to use:** Investment thesis, price targets, buy/sell/hold recommendations.

**Financial example:**
```
Run 1: AAPL analysis → Score 0.72 (bullish on services growth, fair valuation)
Run 2: AAPL analysis → Score 0.68 (bullish on cash generation, slight premium concern)
Run 3: AAPL analysis → Score 0.74 (bullish on ecosystem moat, reasonable valuation)

Consensus: 0.71 (all runs agree on moderately bullish, concerns center on valuation premium)
Variance: Low (0.06 spread) → High confidence in the consensus
```

**Prompt pattern:**
```
Analyze {question} using 3 independent reasoning paths.
For each path:
  - Start fresh (do not reference prior paths)
  - Follow the full analysis workflow
  - Produce an independent score

After all 3 paths:
  - Calculate mean and range of scores
  - Identify common themes across all paths
  - Identify areas of disagreement
  - Final score = mean, confidence = inversely proportional to range
```

**When NOT to use:** Data collection (waste of tokens), simple calculations (deterministic), or when you need speed over accuracy.

---

## Self-Reflection

**What it is:** The model reviews and critiques its own output, checking for errors, inconsistencies, and unsupported claims.

**Best for:** Final quality assurance step in any analysis agent.

**Error catch rate:** ~30% of errors caught during reflection (research finding).

**Prompt pattern (the reflection step):**
```
Before finalizing, critically review your analysis:

1. Score-Data Consistency
   - Read your score. Read your data. Do they agree?
   - If data shows declining margins but score is bullish, something is wrong.

2. Data Support
   - For every claim in your reasoning, identify the specific data point.
   - If a claim has no data backing, either remove it or flag it as assumption.

3. Calculation Verification
   - Re-check one key calculation from scratch.
   - Verify units are consistent (millions vs billions, quarterly vs annual).

4. Temporal Consistency
   - Confirm no data point references events after the analysis date.
   - Confirm comparisons use consistent time periods (YoY or sequential, not mixed).

5. Confidence Check
   - Given the data gaps identified, is your confidence level appropriate?
   - Would you be comfortable defending this score to a portfolio manager?

If any check fails, revise the analysis before presenting.
```

---

## Meta-Prompting (PE2 Framework)

**What it is:** A framework for writing better prompts by including three key components.

**When to use:** When designing any new prompt from scratch.

**The three components:**

1. **Detailed task description** — Not "analyze stocks" but "evaluate the fundamental quality of a company's earnings by assessing cash conversion, accruals quality, and margin sustainability over a 5-year period"

2. **Context specification** — Define the constraints:
   - What data sources are available (MCP tools, subagents)
   - What time period to analyze
   - What benchmarks to compare against
   - What output format is expected
   - What the output will be used for downstream

3. **Step-by-step reasoning template** — Provide the exact structure:
   - Not "analyze step by step" but the actual numbered steps
   - Include what each step should produce
   - Include decision points ("if data is insufficient, then...")

**Financial example:**
```
Task: Evaluate whether AAPL's current P/E premium over the S&P 500 is justified
      by its growth differential and quality metrics.

Context: Use equity_fundamentals tools for financial data, equity_estimates for
         forward metrics. Analysis date: 2024-03-01. Compare against S&P 500
         median and tech sector median. Output will feed into a multi-factor
         stock scoring model.

Steps:
1. Collect AAPL trailing and forward P/E, EPS growth rate (3Y CAGR), ROE, FCF yield
2. Collect S&P 500 and tech sector medians for same metrics
3. Calculate P/E premium: (AAPL P/E / Benchmark P/E - 1) × 100
4. Calculate growth differential: AAPL growth - Benchmark growth
5. Calculate PEG ratios: P/E / EPS growth for AAPL and benchmarks
6. Assess: Is premium justified? PEG < benchmark PEG suggests justified premium
7. Score 0.0-1.0: 0.5 = fairly valued, <0.5 = overvalued, >0.5 = undervalued
```

---

## Few-Shot vs Zero-Shot Decision Guide

| Situation | Recommendation |
|-----------|---------------|
| New prompt, well-structured task | Start zero-shot |
| Output format is inconsistent | Add 1-2 output format examples |
| Reasoning is shallow/wrong | Add 1 detailed reasoning example |
| Edge cases handled poorly | Add examples of edge cases specifically |
| Scoring is inconsistent | Add examples with diverse score ranges (low, mid, high) |

**Few-shot example quality checklist:**
- Covers different sectors (tech, healthcare, industrial, financial)
- Covers different scenarios (strong company, weak company, mixed signals)
- Shows the full reasoning chain, not just input-output
- Demonstrates proper handling of missing data
- All numbers are realistic and internally consistent

---

## Token Optimization

**Key finding:** Performance degrades beyond ~3,000 tokens. Optimal range:
- Data collection: 150-300 words
- Analysis agents: 400-600 words
- Orchestrators: 500-700 words

**Techniques:**
- Use structured formats (tables, numbered lists) over prose
- Tool descriptions: one line per tool, not a paragraph
- Remove redundant instructions (say it once, clearly)
- If a prompt is getting long, split into two agents rather than one bloated prompt
- Use Jinja2 variables for dynamic content instead of maintaining multiple similar prompts
