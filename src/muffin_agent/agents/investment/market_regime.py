"""Stage 2: Market Regime & Top-Down Context."""

import json
import re
from typing import Any

from deepagents import CompiledSubAgent, create_deep_agent
from langchain_core.runnables import RunnableConfig

from muffin_agent.agents.data_collection import (
    create_currency_commodities_data_collection_agent,
    create_economy_macro_data_collection_agent,
    create_etf_index_data_collection_agent,
    create_fama_french_data_collection_agent,
    create_fixed_income_data_collection_agent,
)
from muffin_agent.agents.data_validation import create_data_validation_agent
from muffin_agent.agents.investment.state import TickerAnalysisState
from muffin_agent.config import Configuration
from muffin_agent.prompts import render_template
from muffin_agent.sandbox import get_backend


async def _build_macro_subagents(config: Configuration) -> list[CompiledSubAgent]:
    """Build the macro-focused subagents for market regime analysis.

    Return 5 data collection subagents + 1 data validation subagent, covering
    all macro and regime-relevant data sources.  Excludes company-specific
    subagents (equity-fundamentals, equity-price, etc.) that are irrelevant to
    market-wide regime classification.
    """
    economy_macro_agent = await create_economy_macro_data_collection_agent(config)
    fixed_income_agent = await create_fixed_income_data_collection_agent(config)
    fama_french_agent = await create_fama_french_data_collection_agent(config)
    currency_commodities_agent = (
        await create_currency_commodities_data_collection_agent(config)
    )
    etf_index_agent = await create_etf_index_data_collection_agent(config)
    validation_agent = await create_data_validation_agent(config)

    return [
        CompiledSubAgent(
            name="economy-macro",
            description=(
                "Retrieves macroeconomic data: real GDP growth, CPI/PCE, "
                "unemployment, payrolls, PMI, FOMC minutes (policy tone), "
                "FRED series, UMich sentiment, SLOOS, CLI, money measures. "
                "Primary source for the growth cycle and inflation dimensions."
            ),
            runnable=economy_macro_agent,
        ),
        CompiledSubAgent(
            name="fixed-income",
            description=(
                "Retrieves rates data: Treasury yield curve (3M, 2Y, 5Y, 10Y, 30Y), "
                "EFFR/SOFR, TIPS breakevens (5Y, 10Y), IG/HY credit spreads, "
                "corporate bond yields, and rate forecasts. Primary source for "
                "monetary policy stance and yield curve shape analysis."
            ),
            runnable=fixed_income_agent,
        ),
        CompiledSubAgent(
            name="fama-french",
            description=(
                "Retrieves Fama-French factor data: 5-factor returns "
                "(Mkt-RF, SMB, HML, RMW, CMA) and MOM with trailing 1M/3M/12M "
                "cumulative returns; US portfolio returns by size/value quintile; "
                "breakpoints. Market-wide only — no ticker. Use to assess the "
                "factor regime and calibrate tilt recommendations."
            ),
            runnable=fama_french_agent,
        ),
        CompiledSubAgent(
            name="currency-commodities",
            description=(
                "Retrieves currency and commodity data: FX rates (EUR/USD, "
                "USD/JPY, USD/CNY) as dollar-strength proxies, WTI/Brent crude, "
                "gold (risk sentiment), copper (growth proxy), EIA energy outlook. "
                "Use for commodity inflation signals and risk-appetite context."
            ),
            runnable=currency_commodities_agent,
        ),
        CompiledSubAgent(
            name="etf-index",
            description=(
                "Retrieves ETF and index data: S&P 500 multiples (P/E, P/B, "
                "EV/EBITDA), sector ETF performance (1M, 3M), index snapshots. "
                "If ticker provided, call `etf_equity_exposure` to identify "
                "sector ETFs and infer sector/style. Use for market valuation, "
                "sector rotation, and liquidity signals."
            ),
            runnable=etf_index_agent,
        ),
        CompiledSubAgent(
            name="data-validation",
            description=(
                "Validates collected data against a criterion. Checks "
                "sufficiency, relevance, temporal validity, and consistency. "
                "Returns per-dimension scores (0-1), overall confidence/"
                "relevance scores, identified gaps, and a recommendation "
                "(proceed/collect_more_data/insufficient_data). Use after "
                "data collection, before analysis. Pass the criterion, "
                "analysis date, and all collected data in the task instruction."
            ),
            runnable=validation_agent,
        ),
    ]


async def create_market_regime_agent(config: Configuration):
    """Build the market regime deep agent.

    Create a deep agent that collects macro and fixed-income data, validates
    it, classifies the current regime across 4 dimensions (growth cycle,
    inflation, monetary policy, liquidity/risk appetite), and produces factor
    tilt and positioning guidance.

    ``get_backend`` discovers or creates a sandbox container per conversation
    by ``thread_id`` metadata for Python computations (yield curve slope,
    factor Z-scores, composite indicators).
    """
    subagents = await _build_macro_subagents(config)
    prompt = render_template("market_regime.jinja")
    llm = config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents,
        backend=get_backend,
    )


def _build_task_description(state: dict[str, Any]) -> str:
    """Build the task description string from available state fields.

    Supports 3 context modes:
    - (a) With ticker: TickerAnalysisState — passes ticker + query
    - (b) With sector/country/industry: ScreeningState or custom — passes those fields
    - (c) Query-only: passes only the investment mandate
    """
    parts: list[str] = ["Classify the current macro and market regime."]

    ticker: str | None = state.get("ticker")
    query: str | None = state.get("query")
    sector: str | None = state.get("sector")
    industry: str | None = state.get("industry")
    country: str | None = state.get("country")

    if ticker:
        parts.append(f"Ticker: {ticker}")
    if sector:
        parts.append(f"Sector: {sector}")
    if industry:
        parts.append(f"Industry: {industry}")
    if country:
        parts.append(f"Country/region focus: {country}")
    if query:
        parts.append(f"Investment mandate: {query}")

    if ticker:
        parts.append(
            "Include a ticker_impact section assessing how the current regime "
            f"specifically affects {ticker}."
        )

    return "\n".join(parts)


def _parse_agent_output(output: str) -> dict[str, Any]:
    """Extract and parse the JSON block from the agent's output.

    Return the parsed dict on success, or a minimal error dict if the JSON
    block is missing or malformed.
    """
    match = re.search(
        r"MARKET_REGIME_JSON_START\s*(.*?)\s*MARKET_REGIME_JSON_END",
        output,
        re.DOTALL,
    )
    if not match:
        return {
            "regime_label": "unknown",
            "error": "Agent did not produce a parseable MARKET_REGIME_JSON block",
            "raw_output": output,
        }
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return {
            "regime_label": "unknown",
            "error": f"JSON parse error: {exc}",
            "raw_output": match.group(1),
        }


async def market_regime_node(
    state: TickerAnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Stage 2: Market Regime & Top-Down Context.

    Classifies the current macro and liquidity regime, identifies factor
    tailwinds and headwinds, and sets the top-down frame within which the
    individual company analysis will be interpreted.

    Runs in **parallel** with ``sector_analysis_node`` and
    ``company_analysis_node`` (Group 1).  Its output flows into
    ``forecasting_node`` and ``risk_assessment_node`` (both Group 2) as
    contextual input for macro assumptions and stress-scenario design.

    Also used as a shared context node in the equity screening graph
    (``ScreeningState``) where it runs once before the per-ticker fan-out.
    Accepts both ``TickerAnalysisState`` (with ``ticker``) and
    ``ScreeningState`` (query-only); reads fields via ``.get()`` so missing
    keys degrade gracefully.

    Supported context modes (all optional, combined as available):
        (a) ticker — agent derives sector/style via ``etf_equity_exposure``
        (b) sector / industry / country — passed directly to the agent
        (c) query — investment mandate narrows geographic/style focus

    Outputs (state update):
        market_regime: dict containing:
            - regime_label: str — e.g. "Goldilocks late-cycle"
            - as_of_date: str — YYYY-MM-DD
            - confidence: float — 0.0–1.0
            - dimensions: dict — 4 sub-dicts (growth_cycle, inflation_regime,
              monetary_policy, liquidity_risk_appetite), each with label,
              score, direction, key_indicators
            - factor_assessment: dict — value/quality/momentum/size tilts
            - yield_curve: dict — slope, shape, trend, credit spreads
            - macro_summary: str — 3-4 sentence narrative
            - key_risks: list[str] — macro tail risks
            - recommended_positioning: dict — beta, gross/net, sector/style tilts
            - ticker_impact: dict | absent — regime impact on specific ticker
            - data_sources: list[dict]
            - limitations: list[str]
    """
    configuration = Configuration.from_runnable_config(config)
    agent = await create_market_regime_agent(configuration)

    task = _build_task_description(state)  # type: ignore[arg-type]
    result = await agent.ainvoke({"input": task})

    raw_output: str = (
        result.get("output", "") if isinstance(result, dict) else str(result)
    )
    market_regime = _parse_agent_output(raw_output)

    return {"market_regime": market_regime}
