# Prompt Iteration Guide

How to diagnose prompt issues from Langfuse/LangSmith traces and systematically improve prompts.

---

## Iteration Workflow

### 1. Identify the Problem
Start from observation data in Langfuse or LangSmith:
- **Failed traces:** Errors, timeouts, or empty outputs
- **Low-quality traces:** Hallucinated data, wrong scores, irrelevant analysis
- **Inefficient traces:** Excessive tool retries, token bloat, unnecessary steps
- **Inconsistent traces:** Same input produces different quality outputs across runs

### 2. Categorize the Failure
Map the observation to a failure category:

| Category | Indicators in Traces |
|----------|---------------------|
| **Hallucination** | Numbers in output not present in any tool response |
| **Tool misuse** | Wrong tool called, wrong parameters, unnecessary calls |
| **Excessive retry** | Same tool+args called 2+ times (ToolErrorHandler should prevent this) |
| **Score-data mismatch** | Bullish score with bearish data, or vice versa |
| **Temporal violation** | References to events/data after the analysis date |
| **Token bloat** | >50% of tokens spent on verbose explanations or apologies |
| **Missing analysis** | Skipped required dimensions, no scoring rubric applied |
| **Calculation error** | Wrong arithmetic, mixed units, wrong formula |

### 3. Trace the Root Cause to the Prompt
For each failure category, look at these prompt sections:

| Failure | Prompt Section to Check |
|---------|------------------------|
| Hallucination | Missing no-fabrication clause, missing source grounding requirement |
| Tool misuse | Unclear tool descriptions, missing "Use for..." guidance |
| Excessive retry | Missing or weak error handling block |
| Score-data mismatch | Missing reflection step, missing scoring rubric, no formula-first pattern |
| Temporal violation | Missing temporal anchoring, missing temporal validation step |
| Token bloat | Prompt too long/vague, missing "be concise" directive, too many tools listed |
| Missing analysis | Workflow steps missing, scoring dimensions not defined |
| Calculation error | No formula-first pattern, no sanity ranges, no unit check instruction |

### 4. Apply the Fix
Make a targeted change to the prompt section identified above. Reference the specific patterns in:
- `financial-guardrails.md` for hallucination, calculation, temporal fixes
- `prompting-techniques.md` for reasoning structure fixes
- `data-collection-prompts.md` or `analysis-agent-prompts.md` for structural fixes

### 5. Validate the Fix
Run the same inputs through the updated prompt and compare:
- Does the specific failure still occur?
- Did the fix introduce any regressions?
- Is the token count still within optimal range?

---

## Failure-Fix Reference Table

Quick lookup for common problems:

| Problem | Root Cause | Fix | Reference |
|---------|-----------|-----|-----------|
| Agent invents EPS number | No no-fabrication clause | Add: "NEVER fabricate numbers. State unavailable." | financial-guardrails.md §1 |
| Agent retries failed tool 3x | Error handling block missing/weak | Add standard error handling block verbatim | data-collection-prompts.md |
| Score 0.8 but data shows declining margins | No reflection step | Add Step 5: Reflect with score-data consistency check | analysis-agent-prompts.md |
| Uses Q1 2025 earnings in 2024-12-01 analysis | No temporal anchoring | Add temporal checkpoint pattern | financial-guardrails.md §3 |
| Calculates P/E as 500x (actually 50x) | No sanity ranges | Add metric sanity ranges table | financial-guardrails.md §2 |
| Agent writes 2000-word explanation | Prompt vague, no conciseness directive | Tighten prompt, add "be concise", reduce word count | prompting-techniques.md (token optimization) |
| Wrong tool called for the query | Tool descriptions don't include "Use for..." | Rewrite tool descriptions with clear use-case guidance | data-collection-prompts.md |
| Score varies 0.3-0.8 across runs | No scoring rubric with weights | Add explicit dimension weights summing to 1.0 | analysis-agent-prompts.md |
| Agent mixes quarterly and annual data | No unit/period consistency check | Add: "Verify time periods match before comparing" | financial-guardrails.md §2 |
| Agent says "I apologize" 5 times | Error handling block says "do NOT apologize" but isn't present | Add error handling block | data-collection-prompts.md |
| Missing balance sheet analysis | Tool failed silently, no degradation pattern | Add progressive data degradation pattern | financial-guardrails.md §5 |
| "Revenue is approximately $95B" | Agent estimates instead of citing | Strengthen no-fabrication + source grounding | financial-guardrails.md §1 |
| Agent says "High confidence" with 50% data missing | No confidence breakdown | Add multi-dimensional confidence assessment | financial-guardrails.md §4 |

---

## A/B Testing Protocol

When iterating on a prompt, compare versions systematically:

### Test Setup
1. **Select 10 test cases** covering:
   - 3 large-cap stocks (well-covered, abundant data)
   - 3 mid-cap stocks (moderate coverage)
   - 2 small-cap stocks (sparse data — tests degradation handling)
   - 1 edge case (foreign stock, recently IPO'd, or distressed company)
   - 1 negative case (query that can't be fully answered)

2. **Run each test case 3 times** per prompt version to check consistency

3. **Score on 5 dimensions:**

| Dimension | What to Measure | How to Score |
|-----------|----------------|--------------|
| **Accuracy** | Do numbers match ground truth? | 0-1, automated check against reference data |
| **Consistency** | Same input → same quality output? | Standard deviation of scores across 3 runs |
| **Coverage** | All required dimensions analyzed? | Checklist of expected sections present |
| **Efficiency** | Token usage for equivalent output | Total tokens (lower is better at same quality) |
| **Robustness** | Handles missing data gracefully? | Score on sparse-data test cases |

### Decision Criteria
- **Ship the new version if:** Accuracy improves OR stays same while efficiency improves, AND no dimension regresses >10%
- **Investigate further if:** One dimension improves but another regresses
- **Revert if:** Accuracy drops or robustness degrades

---

## Evaluation Dimensions for Financial Agents

### For Data Collection Agents
| Dimension | Metric |
|-----------|--------|
| **Tool selection accuracy** | % of queries where the right tool was called first |
| **Parameter accuracy** | % of tool calls with correct parameters |
| **Error handling** | % of failures handled gracefully (no retries, brief note) |
| **Summary quality** | Does summary include all key data points? (manual review) |
| **Token efficiency** | Tokens used per successful data retrieval |

### For Analysis Agents
| Dimension | Metric |
|-----------|--------|
| **Score accuracy** | Correlation with expert/ground-truth scores |
| **Score consistency** | Std dev across 3 runs of same input |
| **Reasoning quality** | Every claim backed by data? (manual review) |
| **Calculation accuracy** | % of calculations correct |
| **Temporal compliance** | % of analyses with no future data leakage |
| **Degradation handling** | Quality of output when 30% of data is missing |
| **Confidence calibration** | Does stated confidence match actual accuracy? |

---

## Version Control for Prompts

### File Naming
Prompts live as `.jinja` files in `src/muffin_agent/prompts/`. No version suffixes in filenames — use git history.

### Commit Messages
When changing prompts, commit messages should include:
```
Improve {agent_name} prompt: {what changed and why}

- {specific change 1}
- {specific change 2}
- Motivated by: {Langfuse trace ID or observation}
```

### Tracking Changes in Langfuse
Tag Langfuse traces with prompt version information:
- Use `session_id` or metadata to include prompt git hash
- This allows filtering traces by prompt version for comparison
- Set up in `utils/observability.py` via callback metadata

### Changelog Practice
For significant prompt changes, note in the PR description:
- What failure pattern was observed
- What prompt section was modified
- Expected impact on which evaluation dimensions
- Any A/B test results
