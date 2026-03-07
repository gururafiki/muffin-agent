# Data Collection Agent Prompts

Patterns for writing prompts for agents that retrieve financial data via MCP tools using the ReAct pattern.

## Design Philosophy

Data collection agents are **retrieval-focused, not reasoning-focused**. They should:
- Call the right tools with the right parameters
- Handle failures gracefully without retrying
- Summarize what was retrieved (not analyze it)
- Be concise — the analysis happens elsewhere

## Prompt Template

```
You are a {domain} data collection agent. Your role is to retrieve
{specific data types} using the available tools.

{Context note if tools are market-wide vs ticker-specific}

You have access to the following tools{", grouped by domain:" if 10+ tools}

{Tool listing — see conventions below}

When given a request:
1. Identify what specific {domain} data is needed{" and for which entity" if ticker-specific}.
2. {Search step if tools require lookup — e.g., FRED series ID}
3. Call the appropriate tool(s) with correct parameters.
4. After all tool calls complete, summarize the key findings:
   {domain-specific summary guidance}.
   Note any tool failures briefly.

IMPORTANT — Error handling rules:
- NEVER retry a tool call with the exact same arguments after it fails.
  The system will block duplicate failed calls automatically.
- If a tool fails due to missing credentials or unsupported parameters,
  that tool CANNOT work with these parameters — try different arguments,
  don't repeat calls with the same arguments.
- Instead of retrying with the same arguments, try: (a) a different tool,
  (b) different parameters, or (c) report the data as unavailable.
- Do NOT apologize or explain at length when a tool fails. State what
  failed briefly, then continue with available tools.
```

## Tool Listing Conventions

### Format
Each tool gets one bullet with three parts:
```
- **tool_name**: What it does and what it returns. Use for {specific scenario when this tool is the right choice}.
```

The "Use for..." suffix is critical — it guides the agent's tool selection when handling diverse queries.

### Grouping (for agents with 10+ tools)
Group tools by subdomain with bold headers:
```
**Core indicators:**
- **economy_gdp_real**: Real GDP by country. Use for economic growth assessment.
- **economy_cpi**: Consumer Price Index. Use for inflation assessment.

**Surveys & labor:**
- **economy_survey_nonfarm_payrolls**: Nonfarm payrolls data. Use for labor market strength.
```

### For agents with 2-5 tools
List tools directly without grouping. The simplicity helps keep the prompt short.

### Tool Description Quality
Good tool descriptions answer three questions:
1. **What data does it return?** (not just the tool name restated)
2. **What parameters does it need?** (ticker, date range, series ID — mention non-obvious ones)
3. **When should the agent pick this tool over others?** (the "Use for..." part)

**Bad:** `- **equity_price_quote**: Gets a quote.`
**Good:** `- **equity_price_quote**: Current price, volume, change, market cap, and 52-week range for a ticker. Use for real-time snapshot when historical data is not needed.`

## Search-Then-Fetch Patterns

Some tools require a lookup step before fetching data. Make this explicit:

```
- **economy_fred_search**: Search FRED for series by keyword. Use to find series IDs
  before calling economy_fred_series.
- **economy_fred_series**: Retrieve a FRED data series by series ID (e.g., "GDP",
  "UNRATE"). IMPORTANT: the `symbol` parameter is a FRED series ID, not a stock ticker.
```

The agent must understand the dependency: search first, then fetch with the discovered ID.

Similar patterns apply to:
- BLS series (`economy_survey_bls_search` -> `economy_survey_bls_series`)
- Economic indicators (`economy_available_indicators` -> `economy_indicators`)

## Market-Wide vs Ticker-Specific

If the agent's tools don't require a stock ticker, state this upfront:
```
Unlike equity-specific agents, most of your tools are market-wide and do not
require a stock ticker symbol. They are parameterized by country, date range,
or series IDs.
```

This prevents the agent from trying to pass a ticker where one isn't needed.

## Summary Guidance

The summary instruction should be domain-specific and tell the agent what to highlight:

**Equity fundamentals:** "Summarize key metrics: profitability, leverage, liquidity, growth trends, and notable items."
**Price data:** "Summarize current price, key performance periods, volume trends, and notable support/resistance levels."
**News:** "Summarize most impactful headlines, overall sentiment tone (positive/neutral/negative), notable catalysts or risks, and publication recency."
**Macro:** "Summarize current state of growth, inflation, employment, rates, and any notable trends or risks."

## Common Mistakes

1. **Over-describing tools:** The tool listing should be concise. The agent doesn't need a paragraph per tool.
2. **Missing "Use for..." guidance:** Without it, the agent guesses which tool to use.
3. **Not mentioning parameter quirks:** If a tool's `symbol` parameter means something unusual (FRED series ID, not ticker), say so.
4. **Forgetting the error handling block:** Copy it verbatim — it matches the `ToolErrorHandler` middleware behavior.
5. **Asking the agent to analyze:** Data collection agents retrieve and summarize. Analysis is a separate agent's job.
6. **No summary instruction:** Without it, the agent dumps raw tool output or writes excessively.
