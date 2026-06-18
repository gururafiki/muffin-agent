"""Nassim Taleb persona — compiled subgraph (collect → compute → verdict).

Antifragility + tail risk lens; uses price series for distribution metrics.
See ``warren_buffett.py`` for the canonical reference.
Reference: ``ai-hedge-fund/src/agents/nassim_taleb.py``.

Deliberate omission vs upstream: the upstream agent's
``analyze_black_swan_sentinel`` (a crisis-signal dimension scoring negative-news
ratio + volume spikes + price dislocations, max 4) is NOT ported. Its dominant
input is a company-news negative-sentiment ratio, which would require adding news
collection to this persona's data step; the volume / price-dislocation half is
already implicitly available to the LLM verdict via the tail-risk + vol-regime
evidence. See docs/personas.md / roadmap for the follow-up.
"""

from __future__ import annotations

import logging
import statistics
from typing import Annotated, Any, Literal, cast

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ....model_config import ModelConfiguration
from ....prompts import render_template
from ....sandbox.tools import execute_python
from ....utils.agent_builder import MuffinAgentBuilder
from ...data_collection.utils import get_tools
from ..schemas import AnalystSignal
from ..tools.scoring_helpers import compute_volatility_metrics

logger = logging.getLogger(__name__)
_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence ────────────────────────────────────────────────────────


class NassimTalebTailRisk(BaseModel):
    skewness: float | None
    excess_kurtosis: float | None
    max_drawdown_pct: float | None
    annualized_volatility: float | None
    score: int
    max_score: int
    reasoning: str


class NassimTalebAntifragility(BaseModel):
    net_cash: float | None
    debt_to_assets: float | None
    margin_cv: float | None
    fcf_positive_periods: int
    fcf_total_periods: int
    score: int
    max_score: int
    reasoning: str


class NassimTalebConvexity(BaseModel):
    rd_intensity: float | None
    cash_to_market_cap: float | None
    fcf_yield: float | None
    score: int
    max_score: int
    reasoning: str


class NassimTalebFragility(BaseModel):
    debt_to_equity: float | None
    interest_coverage: float | None
    earnings_cv: float | None
    score: int
    max_score: int
    reasoning: str


class NassimTalebSkinInGame(BaseModel):
    insider_buys: int
    insider_sells: int
    score: int
    max_score: int
    reasoning: str


class NassimTalebVolRegime(BaseModel):
    recent_to_older_vol_ratio: float | None
    score: int
    max_score: int
    reasoning: str


class NassimTalebEvidence(BaseModel):
    tail_risk: NassimTalebTailRisk
    antifragility: NassimTalebAntifragility
    convexity: NassimTalebConvexity
    fragility: NassimTalebFragility
    skin_in_game: NassimTalebSkinInGame
    vol_regime: NassimTalebVolRegime
    annualized_volatility: float | None = None
    skewness: float | None = None
    excess_kurtosis: float | None = None
    max_drawdown_pct: float | None = None
    total_score: float
    max_score: float


class NassimTalebSignal(AnalystSignal):
    agent_id: Literal["nassim_taleb"] = Field(default="nassim_taleb")
    evidence: NassimTalebEvidence


# ── RawData ───────────────────────────────────────────────────────────────────


class NassimTalebRawData(BaseModel):
    """Taleb MCP extraction.  Series oldest -> newest."""

    cash_and_equivalents_series: list[float | None] = Field(default_factory=list)
    total_debt_series: list[float | None] = Field(default_factory=list)
    total_assets_series: list[float | None] = Field(default_factory=list)
    operating_margin_series: list[float | None] = Field(default_factory=list)
    free_cash_flow_series: list[float | None] = Field(default_factory=list)
    revenue_series: list[float | None] = Field(default_factory=list)
    research_and_development_series: list[float | None] = Field(default_factory=list)
    shareholders_equity_series: list[float | None] = Field(default_factory=list)
    ebit_series: list[float | None] = Field(default_factory=list)
    interest_expense_series: list[float | None] = Field(default_factory=list)
    net_income_series: list[float | None] = Field(default_factory=list)
    insider_trades: list[dict[str, Any]] = Field(default_factory=list)
    prices_1y: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Past 252 days of daily OHLCV from equity_price_historical "
            "(provider=yfinance). Each entry must include `close`."
        ),
    )
    market_cap: float | None = None


# ── State ─────────────────────────────────────────────────────────────────────


class NassimTalebInput(TypedDict, total=False):
    ticker: str
    as_of_date: str
    query: str | None


class NassimTalebOutput(TypedDict, total=False):
    persona_signals: list[dict[str, Any]]


class NassimTalebState(AgentState):
    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]
    cash_and_equivalents_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    total_debt_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    total_assets_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    operating_margin_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    free_cash_flow_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    revenue_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    research_and_development_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    shareholders_equity_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    ebit_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    interest_expense_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    net_income_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=False)
    ]
    insider_trades: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=False)
    ]
    prices_1y: Annotated[
        list[dict[str, Any]] | None, OmitFromSchema(input=True, output=False)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=False)]
    evidence: Annotated[
        NassimTalebEvidence | None, OmitFromSchema(input=True, output=False)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers ─────────────────────────────────────────────────────────


def _daily_returns_from_bars(bars: list[dict[str, Any]]) -> list[float]:
    closes = [b.get("close") for b in bars if b.get("close") is not None]
    returns: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev and prev > 0:
            returns.append((closes[i] - prev) / prev)
    return returns


def _score_taleb_tail_risk(
    returns: list[float],
) -> NassimTalebTailRisk:
    metrics = compute_volatility_metrics(returns)
    score = 0
    parts: list[str] = []
    skew = metrics["skewness"]
    kurt = metrics["excess_kurtosis"]
    dd = metrics["max_drawdown_pct"]

    if kurt is not None:
        if kurt > 5:
            score += 2
            parts.append(f"Excess kurtosis {kurt:.1f} — fat tails")
        elif kurt > 2:
            score += 1
    if skew is not None:
        if skew > 0.5:
            score += 2
            parts.append(f"Positive skew {skew:.2f} — upside-heavy distribution")
        elif skew > -0.5:
            score += 1
    if dd is not None:
        if dd > -15:
            score += 2
            parts.append(f"Max drawdown {dd:.1f}% (shallow)")
        elif dd > -30:
            score += 1
    if metrics["annualized_volatility"] is not None:
        parts.append(f"Annualised vol {metrics['annualized_volatility']:.1%}")

    return NassimTalebTailRisk(
        skewness=skew,
        excess_kurtosis=kurt,
        max_drawdown_pct=dd,
        annualized_volatility=metrics["annualized_volatility"],
        score=min(score, 8),
        max_score=8,
        reasoning="; ".join(parts) or "No price data",
    )


def _score_taleb_antifragility(state: NassimTalebState) -> NassimTalebAntifragility:
    cash = [
        v for v in (state.get("cash_and_equivalents_series") or []) if v is not None
    ]
    debt = [v for v in (state.get("total_debt_series") or []) if v is not None]
    assets = [v for v in (state.get("total_assets_series") or []) if v is not None]
    margins = [v for v in (state.get("operating_margin_series") or []) if v is not None]
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]
    market_cap = state.get("market_cap")

    score = 0
    parts: list[str] = []
    net_cash: float | None = None
    de_ratio: float | None = None
    margin_cv: float | None = None
    positives = 0

    if cash and debt:
        net_cash = cash[-1] - debt[-1]
        if net_cash > 0:
            score += 2
            if market_cap and net_cash > 0.20 * market_cap:
                score += 1
                parts.append(f"Net cash {net_cash:,.0f} (>20% of mkt cap)")
            else:
                parts.append(f"Net cash {net_cash:,.0f}")
        elif debt and assets and debt[-1] / assets[-1] < 0.30:
            score += 1
    if debt and assets and assets[-1] and assets[-1] > 0:
        de_ratio = debt[-1] / assets[-1]
        if de_ratio < 0.30:
            score += 2
            parts.append(f"D/assets {de_ratio:.2f}")
        elif de_ratio < 0.50:
            score += 1
    if margins and len(margins) >= 3:
        mean = sum(margins) / len(margins)
        if mean > 0:
            margin_cv = statistics.pstdev(margins) / mean
            if margin_cv < 0.15 and mean > 0.15:
                score += 3
                parts.append(f"Stable margins (CV {margin_cv:.1%}, mean {mean:.1%})")
            elif margin_cv < 0.30 and mean > 0.10:
                score += 2
            elif margin_cv < 0.30:
                score += 1
    if fcf:
        positives = sum(1 for v in fcf if v > 0)
        if positives == len(fcf):
            score += 2
            parts.append("FCF positive every period")
        elif positives >= len(fcf) // 2:
            score += 1

    return NassimTalebAntifragility(
        net_cash=net_cash,
        debt_to_assets=de_ratio,
        margin_cv=margin_cv,
        fcf_positive_periods=positives,
        fcf_total_periods=len(fcf),
        score=min(score, 10),
        max_score=10,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_taleb_convexity(state: NassimTalebState) -> NassimTalebConvexity:
    rd = [
        v for v in (state.get("research_and_development_series") or []) if v is not None
    ]
    revenues = [v for v in (state.get("revenue_series") or []) if v is not None]
    cash = [
        v for v in (state.get("cash_and_equivalents_series") or []) if v is not None
    ]
    fcf = [v for v in (state.get("free_cash_flow_series") or []) if v is not None]
    market_cap = state.get("market_cap")

    score = 0
    parts: list[str] = []
    rd_intensity: float | None = None
    cash_ratio: float | None = None
    fcf_yield: float | None = None

    if rd and revenues and revenues[-1] and revenues[-1] > 0:
        rd_intensity = rd[-1] / revenues[-1]
        if rd_intensity > 0.15:
            score += 3
            parts.append(f"R&D intensity {rd_intensity:.1%} (high convexity)")
        elif rd_intensity > 0.08:
            score += 2
        elif rd_intensity > 0.03:
            score += 1
    if cash and market_cap and market_cap > 0:
        cash_ratio = cash[-1] / market_cap
        if cash_ratio > 0.30:
            score += 3
            parts.append(f"Cash {cash_ratio:.1%} of mkt cap (option value)")
        elif cash_ratio > 0.15:
            score += 2
        elif cash_ratio > 0.05:
            score += 1
    if fcf and market_cap and market_cap > 0:
        fcf_yield = fcf[-1] / market_cap
        if fcf_yield > 0.10:
            score += 2
            parts.append(f"FCF yield {fcf_yield:.1%}")
        elif fcf_yield > 0.05:
            score += 1

    return NassimTalebConvexity(
        rd_intensity=rd_intensity,
        cash_to_market_cap=cash_ratio,
        fcf_yield=fcf_yield,
        score=min(score, 10),
        max_score=10,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_taleb_fragility(state: NassimTalebState) -> NassimTalebFragility:
    debt = [v for v in (state.get("total_debt_series") or []) if v is not None]
    equity = [
        v for v in (state.get("shareholders_equity_series") or []) if v is not None
    ]
    ebit = [v for v in (state.get("ebit_series") or []) if v is not None]
    interest = [
        v for v in (state.get("interest_expense_series") or []) if v is not None
    ]
    net_income = [v for v in (state.get("net_income_series") or []) if v is not None]
    score = 0
    parts: list[str] = []
    de: float | None = None
    cov: float | None = None
    earnings_cv: float | None = None

    if debt and equity and equity[-1] and equity[-1] > 0:
        de = debt[-1] / equity[-1]
        if de < 0.5:
            score += 3
            parts.append(f"Low D/E {de:.2f}")
        elif de < 1.0:
            score += 2
        elif de < 2.0:
            score += 1
        else:
            parts.append(f"Fragile leverage D/E {de:.2f}")
    if ebit and interest and interest[-1] and interest[-1] > 0:
        cov = ebit[-1] / interest[-1]
        if cov > 10:
            score += 2
            parts.append(f"Interest coverage {cov:.1f}x")
        elif cov > 5:
            score += 1
    if net_income and len(net_income) >= 3:
        mean = sum(net_income) / len(net_income)
        if mean > 0:
            earnings_cv = statistics.pstdev(net_income) / mean
            if earnings_cv < 0.20:
                score += 2
                parts.append(f"Stable earnings (CV {earnings_cv:.1%})")
            elif earnings_cv < 0.50:
                score += 1
    return NassimTalebFragility(
        debt_to_equity=de,
        interest_coverage=cov,
        earnings_cv=earnings_cv,
        score=min(score, 8),
        max_score=8,
        reasoning="; ".join(parts) or "Limited data",
    )


def _score_taleb_skin_in_game(state: NassimTalebState) -> NassimTalebSkinInGame:
    insider_trades = state.get("insider_trades") or []
    buys = sum(1 for t in insider_trades if (t.get("transaction_shares") or 0) > 0)
    sells = sum(1 for t in insider_trades if (t.get("transaction_shares") or 0) < 0)
    if buys + sells == 0:
        return NassimTalebSkinInGame(
            insider_buys=0,
            insider_sells=0,
            score=0,
            max_score=4,
            reasoning="No insider activity data",
        )
    ratio = buys / max(sells, 1)
    if ratio > 2:
        return NassimTalebSkinInGame(
            insider_buys=buys,
            insider_sells=sells,
            score=4,
            max_score=4,
            reasoning=f"Strong net buying ({buys}/{sells})",
        )
    if ratio > 0.5:
        return NassimTalebSkinInGame(
            insider_buys=buys,
            insider_sells=sells,
            score=3,
            max_score=4,
            reasoning=f"Net buying ({buys}/{sells})",
        )
    if buys > 0:
        return NassimTalebSkinInGame(
            insider_buys=buys,
            insider_sells=sells,
            score=2,
            max_score=4,
            reasoning=f"Some buying ({buys}/{sells})",
        )
    return NassimTalebSkinInGame(
        insider_buys=buys,
        insider_sells=sells,
        score=1,
        max_score=4,
        reasoning="No insider buying",
    )


def _score_taleb_vol_regime(returns: list[float]) -> NassimTalebVolRegime:
    if len(returns) < 63:
        return NassimTalebVolRegime(
            recent_to_older_vol_ratio=None,
            score=0,
            max_score=6,
            reasoning="Insufficient price history",
        )
    recent = returns[-21:]
    older = returns[-63:-21] if len(returns) >= 63 else returns[:-21]
    recent_vol = statistics.pstdev(recent) if recent else 0
    older_vol = statistics.pstdev(older) if older else 0
    if older_vol == 0:
        return NassimTalebVolRegime(
            recent_to_older_vol_ratio=None,
            score=0,
            max_score=6,
            reasoning="Zero historical vol — anomaly",
        )
    regime = recent_vol / older_vol
    score = 0
    parts: list[str] = []
    if regime < 0.7:
        score = 0
        parts.append(
            f"Vol regime {regime:.2f} — dangerously suppressed (turkey problem)"
        )
    elif regime < 0.9:
        score = 1
    elif regime < 1.3:
        score = 3
        parts.append(f"Normal vol regime {regime:.2f}")
    elif regime < 2.0:
        score = 4
        parts.append(f"Elevated vol regime {regime:.2f} (potentially attractive entry)")
    else:
        score = 2
        parts.append(f"Extreme vol regime {regime:.2f}")
    return NassimTalebVolRegime(
        recent_to_older_vol_ratio=regime,
        score=score,
        max_score=6,
        reasoning="; ".join(parts),
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: NassimTalebState) -> dict[str, Any]:
    prices = state.get("prices_1y") or []
    returns = _daily_returns_from_bars(prices)

    tail_risk = _score_taleb_tail_risk(returns)
    antifragility = _score_taleb_antifragility(state)
    convexity = _score_taleb_convexity(state)
    fragility = _score_taleb_fragility(state)
    skin = _score_taleb_skin_in_game(state)
    vol_regime = _score_taleb_vol_regime(returns)

    total = (
        tail_risk.score
        + antifragility.score
        + convexity.score
        + fragility.score
        + skin.score
        + vol_regime.score
    )
    max_total = (
        tail_risk.max_score
        + antifragility.max_score
        + convexity.max_score
        + fragility.max_score
        + skin.max_score
        + vol_regime.max_score
    )
    evidence = NassimTalebEvidence(
        tail_risk=tail_risk,
        antifragility=antifragility,
        convexity=convexity,
        fragility=fragility,
        skin_in_game=skin,
        vol_regime=vol_regime,
        annualized_volatility=tail_risk.annualized_volatility,
        skewness=tail_risk.skewness,
        excess_kurtosis=tail_risk.excess_kurtosis,
        max_drawdown_pct=tail_risk.max_drawdown_pct,
        total_score=total,
        max_score=max_total,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: NassimTalebState, config: RunnableConfig
) -> dict[str, Any]:
    ticker = state.get("ticker", "")
    as_of_date = state.get("as_of_date", "")
    query = state.get("query")
    evidence = state.get("evidence")
    if evidence is None:
        raise RuntimeError(
            "render_verdict_node called without evidence — "
            "compute_evidence_node must run first"
        )

    llm = ModelConfiguration.get_chat_model_for_role(
        config, "reasoner", schema=NassimTalebSignal
    )
    prompt = render_template(
        "personas_council/personas/nassim_taleb.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        NassimTalebSignal,
        await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage("Render your Taleb verdict now.")]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


_MCP_TOOLS = [
    "equity_fundamental_income",
    "equity_fundamental_balance",
    "equity_fundamental_cash",
    "equity_fundamental_ratios",
    "equity_ownership_insider_trading",
    "equity_price_historical",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)
    builder = (
        MuffinAgentBuilder(primary, name="nassim_taleb_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(NassimTalebState)
        .with_runtime_system_prompt_template(
            "personas_council/personas/nassim_taleb_data_collection.jinja"
        )
        .with_response_format(NassimTalebRawData)
        .with_model_call_limit(run_limit=10, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_nassim_taleb_agent(config: RunnableConfig) -> CompiledStateGraph:
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        NassimTalebState,
        input_schema=NassimTalebInput,
        output_schema=NassimTalebOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=NassimTalebInput,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()
