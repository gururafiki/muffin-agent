---
name: financial-prompt-engineering
description: >
  Use when creating, improving, or iterating on prompts for financial agents
  in the muffin-agent project. Triggers on: "create a prompt", "write agent prompt",
  "improve prompt", "fix prompt", "iterate on prompt", "prompt engineering",
  "scoring rubric", "hallucination prevention", "agent system prompt",
  "design criterion evaluation prompt", "create DCF prompt", or any work on
  .jinja prompt files in src/muffin_agent/prompts/.
---

# Financial Prompt Engineering

Research-backed best practices for crafting institutional-grade prompts for LLM-based financial analysis agents. Synthesized from academic papers (CoT/ToT/GoT in Finance, PE2 Framework, Look-Ahead-Bench), Anthropic/OpenAI prompt engineering guides, and production financial AI systems.

**Use cases:** Create new agent prompts | Improve existing prompts | Iterate based on eval data

**Delivery:** Jinja2 templates (`.jinja`) in `src/muffin_agent/prompts/`, loaded via `render_template()`.

---

## 1. Core Prompt Engineering Principles

### 1.1 Precision Over Generality
Financial prompts must use exact terminology matching regulatory filings, accounting standards, and market conventions. Vague language produces inconsistent, unreliable outputs.

- **Bad:** "How is the company doing financially?"
- **Good:** "Calculate free cash flow as Operating Cash Flow minus Capital Expenditures from the most recent 10-K filing."

### 1.2 Structured Reasoning (Chain-of-Thought)
Break complex analysis into explicit, sequential steps. CoT improves accuracy in financial forecasting by 23-45% (AI in Investment Analysis, 2024). Every analysis prompt should decompose: Problem -> Sub-problems -> Data Requirements -> Analysis -> Synthesis.

### 1.3 Conciseness Matters
Prompt performance degrades beyond ~3,000 tokens. Optimal range: 150-300 words for focused tasks, up to 500-600 for multi-step analysis. Longer is NOT better — be precise and structured, not verbose.

### 1.4 PE2 Meta-Prompting Framework
Three components that improve prompts by ~6% over naive approaches:
1. **Detailed task description** — state exactly what the agent must accomplish
2. **Context specification** — define domain, data sources, constraints, assumptions
3. **Step-by-step reasoning template** — provide the structure for how to think

### 1.5 Zero-Shot First, Few-Shot If Needed
Modern models (GPT-4+, Claude 3.5+) perform well zero-shot. Add 1-3 examples only when zero-shot produces inconsistent format or reasoning. When using few-shot, examples should cover diverse sectors, market conditions, and edge cases.

---

## 2. Financial Domain Guardrails

These are non-negotiable for any financial prompt. See `references/financial-guardrails.md` for detailed patterns.

### 2.1 Hallucination Prevention
LLMs present fabricated financial data with high confidence. Mitigate with:
- **Source grounding:** Every quantitative claim must cite its data source (tool response, filing, period)
- **No-fabrication clause:** "If data is unavailable, state it is unavailable. NEVER estimate, approximate, or fabricate numbers."
- **Structured verification:** Data checks -> Analysis -> Cross-check against benchmarks
- CoT + verification steps reduce hallucination by ~75% vs standard prompting

### 2.2 Calculation Error Prevention
- **Formula-first:** Require the agent to state the formula before calculating: "P/E Ratio = Price / EPS. Price = $175.50, EPS = $6.42, P/E = 27.3x"
- **Sanity ranges:** Include plausible ranges for key metrics (P/E: 5-100x, Revenue Growth: -50% to +200%, Debt/Equity: 0-10x, Operating Margin: -100% to +80%)
- **Unit consistency:** Explicitly require checking that units match (millions vs billions, quarterly vs annual)
- **Deterministic computation:** For complex math, instruct the agent to write Python code rather than doing arithmetic in natural language

### 2.3 Look-Ahead Bias Prevention
LLMs have training data from the future relative to any historical analysis date. This is the #1 source of unreliable financial backtests.
- **Point-in-time anchoring:** Every analysis must declare: "Analyze as of [DATE]. Use only data available before this date."
- **Temporal checkpoints:** Add an explicit validation step: "Verify no data point references events after [DATE]"
- **Available data declaration:** List what filings/reports would have been published by the analysis date
- 54% of finance professionals rate look-ahead bias as "extremely critical" (Look-Ahead-Bench, 2025)

### 2.4 Confidence Calibration
LLM confidence scores are poorly calibrated (ECE 0.12-0.40). Mitigate with multi-dimensional breakdown:
- **Data sufficiency:** Were all required data points obtained?
- **Data recency:** How current is the data? (days/weeks/months old)
- **Source quality:** Official filings > analyst estimates > news > social media
- **Coverage gaps:** What material data is missing, and how does it affect the conclusion?

### 2.5 Progressive Data Degradation
Financial data is often incomplete. Agents must handle this gracefully:
- Acknowledge specific gaps (don't silently skip missing data)
- Proceed with available data, explicitly noting what was used
- Mark limitations in output: "Analysis limited by: {specific gaps}"
- Reduce confidence score proportionally to data gaps
- NEVER fill gaps with fabricated estimates

---

## 3. Prompting Techniques for Financial Agents

Choose the right technique for the task. See `references/prompting-techniques.md` for detailed patterns with financial examples.

| Technique | Best For | Accuracy Gain | When to Use |
|-----------|----------|---------------|-------------|
| **ReAct** | Data retrieval | Baseline | Data collection agents that call tools |
| **Chain-of-Thought** | Linear analysis | +23-45% | Earnings analysis, ratio calculation, compliance |
| **Tree-of-Thought** | Multi-path exploration | Moderate | Multiple valuation methods, scenario analysis |
| **Graph-of-Thought** | Multi-dimensional synthesis | +15-25%, -25-30% hallucination | Multi-factor stock scoring, criteria synthesis |
| **Self-Consistency** | High-stakes decisions | Better than single-path | Investment recommendations, price targets |
| **Self-Reflection** | Quality assurance | Catches ~30% errors | Final validation step in any analysis agent |

**Default recommendations:**
- Data collection -> ReAct (built into `create_agent`)
- Single-factor analysis -> Chain-of-Thought
- Multi-factor scoring -> Graph-of-Thought
- Final step of any analysis agent -> Self-Reflection

---

## 4. Multi-Dimensional Analysis Framework

For analysis agents that score stocks, use explicit multi-dimensional scoring. This is the institutional standard.

### Scoring Architecture
```
Overall Score (0.0-1.0) = weighted combination of dimension scores

Each dimension:
  - Explicit weight (all weights sum to 1.0)
  - Sub-criteria with individual scores
  - Data points backing each sub-criterion score
  - Reasoning chain from data -> sub-score -> dimension score -> overall score
```

### Standard Dimensions (customize per agent)
| Dimension | Weight Range | Sub-Criteria Examples |
|-----------|-------------|----------------------|
| **Quality** | 0.20-0.35 | Profitability margins, ROE/ROIC, cash conversion, competitive moats |
| **Growth** | 0.15-0.25 | Revenue CAGR, EPS growth, market expansion, pipeline |
| **Valuation** | 0.20-0.30 | P/E vs peers/history, EV/EBITDA, FCF yield, PEG ratio |
| **Risk** | 0.15-0.25 | Leverage, liquidity, concentration, regulatory, volatility |
| **Catalyst/Momentum** | 0.05-0.15 | Estimate revisions, insider activity, upcoming events, sentiment |

### Scoring Discipline
- Score must reference specific numbers: "ROE of 24.3% vs sector median 15.1% -> Quality sub-score: 0.78"
- Never assign scores based on narrative alone
- Explain premium/discount to fair value quantitatively
- Flag when data is insufficient to score a dimension (reduce weight, redistribute)

---

## 5. Prompt Structure Template

### Data Collection Agents
See `references/data-collection-prompts.md` for the full template and examples.

```
ROLE: "You are a {domain} data collection agent. Your role is to retrieve {specific data types}."
TOOLS: List each tool with: bold name, what it does, "Use for {specific scenario}"
WORKFLOW: Numbered steps — identify needs, call tools, summarize findings
ERROR HANDLING: Standard block (see reference file)
```

### Analysis Agents
See `references/analysis-agent-prompts.md` for the full template and examples.

```
ROLE + OBJECTIVE: Clear analytical goal
AVAILABLE DATA SOURCES: Subagents or tools with capabilities
WORKFLOW:
  Step 1 — Plan: Define data needs from the analytical question
  Step 2 — Collect: Gather data via subagents/tools
  Step 3 — Validate: Check sufficiency, relevance, temporal correctness, completeness
  Step 4 — Analyze: Score with explicit rubric, cite data, show reasoning chain
  Step 5 — Reflect: Verify score-data consistency, logical coherence, confidence level
OUTPUT FORMAT: Structured (score, confidence, data_used, reasoning, limitations)
```

---

## 6. Creating a New Prompt — Workflow

1. **Identify agent type** (data collection / analysis / orchestrator) -> read the right reference
2. **Define single responsibility** — one sentence describing what this agent does and nothing else
3. **Choose prompting technique** from Section 3
4. **Draft the prompt** following the template from Section 5
5. **Apply financial guardrails** from Section 2 — check each one
6. **Add scoring rubric** (analysis agents) with explicit weights and data requirements
7. **Add validation + reflection steps** (analysis agents)
8. **Check token count** — aim for sweet spot (150-300 words data collection, 400-600 analysis, 500-700 orchestrators)
9. **Save as `.jinja`** in `src/muffin_agent/prompts/`
10. **Test with edge cases** — missing data, stale data, ambiguous queries

---

## 7. Improving an Existing Prompt

Read the prompt, then evaluate against these checklists.

### Prompt Quality Checklist
- [ ] Task description is specific and unambiguous?
- [ ] Uses precise financial terminology (not vague language)?
- [ ] Reasoning steps are explicit (not "analyze the data")?
- [ ] Output format is structured and defined?
- [ ] Token count is within optimal range?

### Financial Guardrails Checklist
- [ ] Temporal anchoring ("as of [DATE]") present?
- [ ] Source grounding required for all claims?
- [ ] No-fabrication clause included?
- [ ] Formula-first calculation pattern used?
- [ ] Sanity ranges for key metrics specified?
- [ ] Confidence breakdown is multi-dimensional?
- [ ] Data degradation handled gracefully?

### Anti-Patterns to Fix (see `references/financial-guardrails.md`)
- Vague requests -> specify exact metrics, date ranges, filing types
- Unbounded time ranges -> explicit start/end dates
- Missing error handling -> add standard error handling block
- Narrative-only scoring -> require data-backed scores with formulas
- No temporal validation -> add point-in-time check step
- Excessive prompt length -> tighten, remove redundancy, use structured formats

---

## 8. Iterating Based on Evaluation Data

When Langfuse/LangSmith traces reveal issues, diagnose and fix systematically. See `references/prompt-iteration-guide.md` for the full workflow.

### Quick Diagnosis Table
| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Repeated tool retries | Weak error handling | Add/strengthen error handling block |
| Score contradicts data | No reflection, weak rubric | Add reflection step, explicit scoring criteria |
| Fabricated numbers | No source grounding | Add no-fabrication clause + source citation requirement |
| Future data leakage | No temporal anchoring | Add point-in-time anchor + temporal validation step |
| Token bloat / verbosity | Prompt too vague or long | Tighten instructions, add "be concise" directive |
| Tool misuse | Unclear tool descriptions | Improve per-tool "Use for..." guidance |
| Inconsistent scores | No dimension weights | Add explicit weights summing to 1.0 |
| Low-quality reasoning | No CoT structure | Add numbered reasoning steps with formula-first pattern |
| Overconfident with sparse data | No confidence breakdown | Add multi-dimensional confidence (sufficiency/recency/quality/coverage) |

For systematic comparison of prompt versions, follow the A/B testing protocol in `references/prompt-iteration-guide.md`.

---

## 9. Unbiasing Architecture (for criterion/criteria evaluation agents)

Design agents to prevent cognitive bias from leaking into evaluations. See `references/analysis-agent-prompts.md` (Planned Agent Templates) for implementation patterns.

1. **Data needs definition:** Agent should NOT know which subagents exist — define needs from the criterion alone, then map to available subagents
2. **Criterion evaluation:** Agent receives ONLY the criterion text + collected data (no ticker, no company name)
3. **Reflection:** Agent sees ONLY data + reasoning (ticker-blind review)
4. **Synthesis:** Agent combines ONLY criterion scores + reasoning (ticker-blind synthesis)

---

## Reference Files

| File | Content |
|------|---------|
| [`references/data-collection-prompts.md`](references/data-collection-prompts.md) | Template, tool listing conventions, grouping, search-then-fetch patterns, examples |
| [`references/analysis-agent-prompts.md`](references/analysis-agent-prompts.md) | 5-step workflow, scoring rubrics, output formats, planned agent templates, unbiasing |
| [`references/prompting-techniques.md`](references/prompting-techniques.md) | CoT, ToT, GoT, Self-Consistency, Self-Reflection, Meta-Prompting with financial examples |
| [`references/financial-guardrails.md`](references/financial-guardrails.md) | Hallucination prevention, calculation errors, look-ahead bias, confidence calibration, anti-patterns |
| [`references/prompt-iteration-guide.md`](references/prompt-iteration-guide.md) | Langfuse/LangSmith diagnosis workflow, failure-fix mappings, A/B testing, versioning |
