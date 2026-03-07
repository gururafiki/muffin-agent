# Analysis Agent Prompts

Patterns for writing prompts for agents that analyze financial data, score stocks, evaluate criteria, and synthesize multi-source assessments.

## Design Philosophy

Analysis agents are **reasoning-focused**. They must:
- Follow a structured multi-step workflow (not freeform reasoning)
- Validate data before analyzing it
- Produce quantitative scores backed by specific data points
- Reflect on their own output for consistency
- Handle incomplete data gracefully without fabrication

## The 5-Step Analysis Workflow

Every analysis agent should follow this pattern. Adapt the specifics to the agent's purpose.

### Step 1 — Plan
Define what data is needed to answer the analytical question. Be specific:
- What metrics/data points are required?
- What time period is relevant?
- What comparison benchmarks are needed (peers, historical, sector)?
- What constitutes sufficient data to produce a meaningful assessment?

**Key instruction:** "Write your data collection plan before proceeding."

### Step 2 — Collect
Gather data through subagents or tools. Instructions should:
- Specify how to delegate (e.g., `task()` tool with subagent name)
- Require specific, targeted requests ("Get AAPL FY2023 income statement, balance sheet, and cash flow" not "Get AAPL financial data")
- Allow parallel collection when data sources are independent
- Require saving collected data for reference

### Step 3 — Validate
Check collected data before analyzing. Four validation dimensions:

| Dimension | Question | Action if Fails |
|-----------|----------|-----------------|
| **Sufficiency** | Enough data to meaningfully answer the query? | Collect more or note gap |
| **Relevance** | Does data address what the query actually asks? | Re-collect with corrected request |
| **Temporal correctness** | All data predates the analysis date? | Remove future-dated data, flag it |
| **Completeness** | Any critical gaps preventing fair assessment? | Collect additional data or note limitation |

**Key instruction:** "If validation fails, collect additional data. If data cannot be obtained, note the gaps explicitly and proceed with available data."

### Step 4 — Analyze
Produce the scored assessment. The prompt must specify:

**Scoring rubric:**
```
Score each dimension:
1. {Dimension 1} (weight: 0.XX):
   - Sub-criteria: {list}
   - Data required: {list}
   - Score 0.0-1.0 based on: {specific criteria}

2. {Dimension 2} (weight: 0.XX):
   ...

Overall score = weighted combination of dimension scores
```

**Reasoning requirements:**
- Reference actual numbers from collected data
- Show the formula before calculating
- Explain why the score is where it is (not just what the number is)
- Compare to relevant benchmarks (peer median, historical average, sector norm)

**Example instruction:** "Your score must reference specific data: 'ROE of 24.3% vs sector median 15.1% indicates superior profitability -> Quality score: 0.78.' Never score based on narrative alone."

### Step 5 — Reflect
Self-check before finalizing. Include these checks:

```
Before finalizing, verify:
- Score-data consistency: Does the score direction match what the data shows?
- Data support: Is every claim in the reasoning backed by a specific data point?
- Confidence appropriateness: Is the assessment hedged given data quality/gaps?
- Logical coherence: Any contradictions between data, reasoning, and score?
- Temporal consistency: Does the analysis use only data available as of [DATE]?
- Sanity check: Are calculated metrics within plausible ranges?

If reflection reveals issues, revise before presenting the final result.
```

Research shows self-reflection catches ~30% of errors before they reach the output.

## Output Format Template

```
## {Analysis Type}: {TICKER or Subject}

**Score**: X.X / 1.0 ({interpretation})
**Confidence**: {High/Medium/Low} ({brief justification})
**Relevance**: {High/Medium/Low}

### Scoring Breakdown
| Dimension | Weight | Score | Key Data |
|-----------|--------|-------|----------|
| {Dim 1}  | 0.XX   | 0.XX  | {specific metric} |
| {Dim 2}  | 0.XX   | 0.XX  | {specific metric} |

### Reasoning
{Detailed reasoning with specific numbers, formulas shown, comparisons made}

### Data Used
- {specific data point 1 with source and period}
- {specific data point 2 with source and period}

### Limitations
- {specific gap 1 and its impact on the assessment}
- {specific gap 2 and its impact on the assessment}
```

## Orchestrator Agent Pattern

Orchestrators delegate to subagents and synthesize results. Additional prompt elements:

### Subagent Listing
```
You have access to {N} subagents:
- **subagent-name**: {What data/analysis it provides}. Use for {when to delegate to it}.
```

### Delegation Instruction
```
Use the `task` tool to delegate to subagents. Be specific in task descriptions —
include the ticker, exact data to retrieve, and any date constraints.

Example: task(subagent_type="equity-fundamentals",
              description="Get AAPL FY2023 10-K: income statement, balance sheet,
              cash flow, key ratios (ROE, ROIC, Debt/Equity, Current Ratio)")
```

### Synthesis Step
After collecting subagent outputs, the orchestrator must:
- Cross-reference data across subagents for consistency
- Identify convergence/divergence in signals
- Weight subagent contributions based on relevance to the query
- Produce a unified assessment, not just concatenated summaries

## Planned Agent Templates

### Criterion Evaluation Agent
Evaluates a single criterion against collected data. Key design decisions:

```
You are a criterion evaluation agent. You evaluate a single investment
criterion using collected financial data.

**Input:** A criterion (e.g., "profitability trend") and a data package.

**Unbiasing rule:** You do NOT know the company name, ticker, or sector.
Evaluate purely based on the data and criterion provided.

Workflow:
1. Parse the criterion — what specifically must be assessed?
2. Identify what data from the package is relevant to this criterion
3. Evaluate using Chain-of-Thought:
   a. State the assessment framework (what constitutes strong/weak for this criterion)
   b. Map data points to the framework
   c. Calculate/compare relevant metrics (formula-first)
   d. Score 0.0-1.0 with explicit justification
4. Assess confidence (data sufficiency for this specific criterion)
5. Reflect: Does the score match the data? Any logical gaps?

Output: score, confidence, reasoning (with data citations), limitations
```

### Criteria Evaluation Agent (Orchestrator)
Defines and evaluates multiple criteria for a stock.

```
You are a criteria evaluation agent. You define investment criteria
relevant to a given context and orchestrate their evaluation.

Workflow:
1. Collect context data (industry, sector, business model, market cap)
2. Define 5-10 evaluation criteria, each with:
   - Description of what it measures
   - Relevance weight (sum to 1.0)
   - What data is needed to evaluate it
3. For each criterion, delegate to criterion evaluation subagent
4. Synthesize results:
   - Weighted score from criterion scores
   - Identify agreements/conflicts across criteria
   - Flag criteria with low confidence
5. Reflect on synthesis — is the overall score internally consistent?

**Unbiasing rule:** When synthesizing, do NOT reference the ticker.
Combine only criterion scores and reasoning.
```

### DCF Valuation Agent
Requires especially strong calculation guardrails.

```
Key prompt elements for DCF:
- Formula-first for every calculation step
- Explicit assumption declaration (growth rate, discount rate, terminal value method)
- Sensitivity analysis required (at minimum: growth rate +/- 2%, discount rate +/- 1%)
- Sanity check: implied price vs current price within reasonable range (0.3x-3.0x)
- WACC derivation must show all components (risk-free rate, beta, equity risk premium, cost of debt, tax rate, capital structure)
- Terminal value methodology must be stated and justified (Gordon growth vs exit multiple)
- All dollar amounts must specify units (millions/billions) and currency
```

### Specialized Analysis Agents (Technical, Fundamental, Macro, Sentiment)
These agents focus on one analytical dimension. Key differences from the general pattern:

**Technical Analysis:**
- Score based on indicator signals (RSI, MACD, Bollinger Bands, volume)
- Timeframe must be explicit (daily, weekly, monthly)
- Include trend identification (uptrend/downtrend/sideways)
- Avoid opinion-based language — describe what indicators show

**Fundamental Analysis:**
- Score based on quality, growth, profitability, balance sheet health
- Normalize for one-time items (restructuring charges, legal settlements)
- Compare to relevant peers (same sector, similar market cap)
- Calculate metrics from raw data rather than using pre-calculated ratios when possible

**Macro Analysis:**
- Focus on how macro environment affects the specific company/sector
- Include rate environment impact (discount rates, borrowing costs)
- Assess cyclical positioning
- Flag regime changes (monetary policy shifts, trade policy changes)

**Sentiment Analysis:**
- Distinguish between news sentiment, social sentiment, and analyst sentiment
- Weight by recency (recent > old)
- Flag potential manipulation signals (unusual volume of positive/negative coverage)
- Include relevance filtering (company-specific news vs general market noise)
