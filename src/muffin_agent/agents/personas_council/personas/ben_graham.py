"""Ben Graham persona — compiled subgraph (collect → compute → verdict).

Three-node :class:`StateGraph` subgraph implementing the persona pattern.
See ``warren_buffett.py`` for the canonical reference implementation.

Three composite scorers: earnings stability (max 4), financial strength
(max 5), valuation via NCAV + Graham Number (max 6).  Total max = 15.

Reference (upstream): ``ai-hedge-fund/src/agents/ben_graham.py``.
"""

from __future__ import annotations

import logging
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
from ..tools.scoring_helpers import (
    compute_graham_number,
    compute_ncav_per_share,
)

logger = logging.getLogger(__name__)

_LLM_RETRY = RetryPolicy(max_attempts=2)


# ── Typed sub-evidence Pydantics ──────────────────────────────────────────────


class BenGrahamEarningsStability(BaseModel):
    """EPS positivity + growth signal across the available history."""

    positive_periods: int
    total_periods: int
    eps_latest: float | None
    eps_oldest: float | None
    eps_grew: bool
    score: int
    max_score: int
    reasoning: str


class BenGrahamFinancialStrength(BaseModel):
    """Current ratio, debt ratio, dividend record."""

    current_ratio: float | None
    debt_to_assets_ratio: float | None
    dividends_paid_years: int
    dividends_window_years: int
    score: int
    max_score: int
    reasoning: str


class BenGrahamValuation(BaseModel):
    """NCAV check + Graham Number margin of safety."""

    score: int
    max_score: int
    reasoning: str


class BenGrahamEvidence(BaseModel):
    """Graham-specific precomputed evidence."""

    earnings_stability: BenGrahamEarningsStability
    financial_strength: BenGrahamFinancialStrength
    valuation: BenGrahamValuation

    ncav_per_share: float | None = Field(
        default=None,
        description="Net Current Asset Value per share = (CA - liabilities) / shares.",
    )
    graham_number: float | None = Field(
        default=None,
        description="sqrt(22.5 x EPS x BVPS); classic Graham intrinsic estimate.",
    )
    current_price: float | None = Field(
        default=None,
        description="market_cap / outstanding_shares (per-share market price).",
    )
    margin_of_safety_graham_pct: float | None = Field(
        default=None,
        description=("(graham_number - current_price) / current_price, in percent."),
    )
    is_net_net: bool = Field(
        default=False,
        description="True when NCAV total > market cap — classic Graham net-net.",
    )

    total_score: float
    max_score: float


class BenGrahamSignal(AnalystSignal):
    """Narrowed signal with typed Graham evidence."""

    agent_id: Literal["ben_graham"] = Field(default="ben_graham")
    evidence: BenGrahamEvidence


# ── RawData (output of the data_react sub-agent) ──────────────────────────────


class BenGrahamRawData(BaseModel):
    """Structured MCP extraction — the data_react sub-agent's response_format.

    All time series are **oldest → newest** order.  Latest scalar values are
    pulled from the most-recent annual snapshot.
    """

    eps_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "Annual diluted EPS, oldest -> newest, from equity_fundamental_income."
        ),
    )
    dividends_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "Annual dividends_and_other_cash_distributions "
            "(NEGATIVE = cash outflow to shareholders), oldest -> newest, "
            "from equity_fundamental_cash."
        ),
    )
    current_assets_latest: float | None = Field(
        default=None,
        description="Latest annual current_assets, from equity_fundamental_balance.",
    )
    current_liabilities_latest: float | None = Field(
        default=None,
        description=(
            "Latest annual current_liabilities, from equity_fundamental_balance."
        ),
    )
    total_assets_latest: float | None = Field(
        default=None,
        description="Latest annual total_assets, from equity_fundamental_balance.",
    )
    total_liabilities_latest: float | None = Field(
        default=None,
        description=(
            "Latest annual total_liabilities, from equity_fundamental_balance."
        ),
    )
    outstanding_shares_latest: float | None = Field(
        default=None,
        description=(
            "Latest annual diluted shares outstanding, from "
            "equity_fundamental_balance OR equity_fundamental_income."
        ),
    )
    book_value_per_share_latest: float | None = Field(
        default=None,
        description=(
            "Latest annual book value per share. If not directly reported, "
            "compute as shareholders_equity / outstanding_shares."
        ),
    )
    market_cap: float | None = Field(
        default=None,
        description="Latest market capitalisation, from equity_historical_market_cap.",
    )


# ── State schema ──────────────────────────────────────────────────────────────


class BenGrahamInput(TypedDict, total=False):
    """Public input contract — what the council provides to this persona."""

    ticker: str
    as_of_date: str
    query: str | None


class BenGrahamOutput(TypedDict, total=False):
    """Public output contract for the Ben Graham persona subgraph."""

    persona_signals: list[dict[str, Any]]


class BenGrahamState(AgentState):
    """Internal state schema for the Ben Graham persona subgraph."""

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    as_of_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    query: Annotated[str | None, OmitFromSchema(input=False, output=True)]

    eps_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    dividends_series: Annotated[
        list[float | None] | None, OmitFromSchema(input=True, output=True)
    ]
    current_assets_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=True)
    ]
    current_liabilities_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=True)
    ]
    total_assets_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=True)
    ]
    total_liabilities_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=True)
    ]
    outstanding_shares_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=True)
    ]
    book_value_per_share_latest: Annotated[
        float | None, OmitFromSchema(input=True, output=True)
    ]
    market_cap: Annotated[float | None, OmitFromSchema(input=True, output=True)]

    evidence: Annotated[
        BenGrahamEvidence | None, OmitFromSchema(input=True, output=True)
    ]
    persona_signals: Annotated[list[dict], OmitFromSchema(input=True, output=False)]


# ── Composite scorers (pure Python) ───────────────────────────────────────────


def _score_graham_earnings_stability(
    eps_series: list[float | None] | None,
) -> BenGrahamEarningsStability:
    """Score Graham's earnings stability (max 4)."""
    if not eps_series:
        return BenGrahamEarningsStability(
            positive_periods=0,
            total_periods=0,
            eps_latest=None,
            eps_oldest=None,
            eps_grew=False,
            score=0,
            max_score=4,
            reasoning="No EPS history.",
        )
    eps = [v for v in eps_series if v is not None]
    if len(eps) < 2:
        return BenGrahamEarningsStability(
            positive_periods=0,
            total_periods=len(eps),
            eps_latest=eps[-1] if eps else None,
            eps_oldest=eps[0] if eps else None,
            eps_grew=False,
            score=0,
            max_score=4,
            reasoning=f"Insufficient EPS history ({len(eps)} valid periods).",
        )

    positives = sum(1 for e in eps if e > 0)
    score = 0
    parts: list[str] = []
    if positives == len(eps):
        score += 3
        parts.append("EPS positive in every period")
    elif positives >= int(len(eps) * 0.8):
        score += 2
        parts.append(f"EPS positive in {positives}/{len(eps)} periods")
    else:
        parts.append(f"EPS negative in {len(eps) - positives}/{len(eps)} periods")

    grew = eps[-1] > eps[0]
    if grew:
        score += 1
        parts.append("EPS grew from earliest to latest period")
    else:
        parts.append("EPS did not grow over the window")

    return BenGrahamEarningsStability(
        positive_periods=positives,
        total_periods=len(eps),
        eps_latest=eps[-1],
        eps_oldest=eps[0],
        eps_grew=grew,
        score=score,
        max_score=4,
        reasoning="; ".join(parts),
    )


def _score_graham_financial_strength(
    current_assets: float | None,
    current_liabilities: float | None,
    total_assets: float | None,
    total_liabilities: float | None,
    dividends_series: list[float | None] | None,
) -> BenGrahamFinancialStrength:
    """Score Graham's financial-strength dimension (max 5)."""
    score = 0
    parts: list[str] = []
    cr: float | None = None
    de: float | None = None

    if current_assets is not None and current_liabilities and current_liabilities > 0:
        cr = current_assets / current_liabilities
        if cr >= 2.0:
            score += 2
            parts.append(f"Current ratio {cr:.2f} (>=2 strong)")
        elif cr >= 1.5:
            score += 1
            parts.append(f"Current ratio {cr:.2f} (moderate)")
        else:
            parts.append(f"Current ratio {cr:.2f} (weak)")
    else:
        parts.append("Current ratio unavailable")

    if total_assets and total_liabilities is not None and total_assets > 0:
        de = total_liabilities / total_assets
        if de < 0.5:
            score += 2
            parts.append(f"Debt ratio {de:.2f} (conservative)")
        elif de < 0.8:
            score += 1
            parts.append(f"Debt ratio {de:.2f} (acceptable)")
        else:
            parts.append(f"Debt ratio {de:.2f} (high)")
    else:
        parts.append("Debt ratio unavailable")

    divs = [d for d in (dividends_series or []) if d is not None]
    paid_years = sum(1 for d in divs if d < 0)
    if divs:
        if paid_years >= len(divs) // 2 + 1:
            score += 1
            parts.append(f"Dividends paid in {paid_years}/{len(divs)} years (majority)")
        elif paid_years > 0:
            parts.append(f"Dividends paid in {paid_years}/{len(divs)} years (minority)")
        else:
            parts.append("No dividends paid")
    else:
        parts.append("No dividend data")

    return BenGrahamFinancialStrength(
        current_ratio=cr,
        debt_to_assets_ratio=de,
        dividends_paid_years=paid_years,
        dividends_window_years=len(divs),
        score=score,
        max_score=5,
        reasoning="; ".join(parts),
    )


def _score_graham_valuation(
    current_assets: float | None,
    total_liabilities: float | None,
    outstanding_shares_latest: float | None,
    eps_latest: float | None,
    bvps_latest: float | None,
    market_cap: float | None,
) -> tuple[BenGrahamValuation, dict[str, Any]]:
    """Graham valuation: NCAV + Graham Number MoS (max 6).

    Returns the scoring + a dict of extra evidence facts (ncav, graham number,
    margin of safety, is_net_net) that get composed into the full Evidence.
    """
    score = 0
    parts: list[str] = []
    extras: dict[str, Any] = {
        "ncav_per_share": None,
        "graham_number": None,
        "current_price": None,
        "margin_of_safety_graham_pct": None,
        "is_net_net": False,
    }

    if (
        outstanding_shares_latest is None
        or outstanding_shares_latest <= 0
        or market_cap is None
        or market_cap <= 0
    ):
        parts.append("Cannot compute Graham valuation (missing market data)")
        return (
            BenGrahamValuation(score=0, max_score=6, reasoning="; ".join(parts)),
            extras,
        )

    current_price = market_cap / outstanding_shares_latest
    extras["current_price"] = current_price

    ncav_per_share = compute_ncav_per_share(
        current_assets, total_liabilities, outstanding_shares_latest
    )
    extras["ncav_per_share"] = ncav_per_share

    if ncav_per_share is not None:
        ncav_total = ncav_per_share * outstanding_shares_latest
        if ncav_total > market_cap:
            score += 4
            extras["is_net_net"] = True
            parts.append("NCAV > market cap — classic Graham net-net")
        elif ncav_per_share >= current_price * 0.67:
            score += 2
            parts.append("NCAV >= 67% of price — moderate net-net discount")

    graham_number = compute_graham_number(eps_latest, bvps_latest)
    extras["graham_number"] = graham_number
    if graham_number is not None and current_price > 0:
        mos = (graham_number - current_price) / current_price
        extras["margin_of_safety_graham_pct"] = mos * 100
        if mos > 0.5:
            score += 3
            parts.append(f"Graham Number margin of safety {mos:.1%} (>=50%)")
        elif mos > 0.2:
            score += 1
            parts.append(f"Graham Number margin of safety {mos:.1%}")
        else:
            parts.append(f"Graham Number margin of safety {mos:.1%} (low)")
    else:
        parts.append("Graham Number unavailable (need positive EPS and BVPS)")

    return (
        BenGrahamValuation(score=score, max_score=6, reasoning="; ".join(parts)),
        extras,
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────


def compute_evidence_node(state: BenGrahamState) -> dict[str, Any]:
    """Deterministic compute step — read RawData, compose BenGrahamEvidence."""
    earnings_stability = _score_graham_earnings_stability(state.get("eps_series"))
    financial_strength = _score_graham_financial_strength(
        state.get("current_assets_latest"),
        state.get("current_liabilities_latest"),
        state.get("total_assets_latest"),
        state.get("total_liabilities_latest"),
        state.get("dividends_series"),
    )
    valuation, extras = _score_graham_valuation(
        state.get("current_assets_latest"),
        state.get("total_liabilities_latest"),
        state.get("outstanding_shares_latest"),
        (state.get("eps_series") or [None])[-1] if state.get("eps_series") else None,
        state.get("book_value_per_share_latest"),
        state.get("market_cap"),
    )

    total = earnings_stability.score + financial_strength.score + valuation.score
    max_total = (
        earnings_stability.max_score
        + financial_strength.max_score
        + valuation.max_score
    )

    evidence = BenGrahamEvidence(
        earnings_stability=earnings_stability,
        financial_strength=financial_strength,
        valuation=valuation,
        ncav_per_share=extras["ncav_per_share"],
        graham_number=extras["graham_number"],
        current_price=extras["current_price"],
        margin_of_safety_graham_pct=extras["margin_of_safety_graham_pct"],
        is_net_net=extras["is_net_net"],
        total_score=total,
        max_score=max_total,
    )
    return {"evidence": evidence}


async def render_verdict_node(
    state: BenGrahamState, config: RunnableConfig
) -> dict[str, Any]:
    """Single LLM call: render BenGrahamSignal with structured output."""
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
        config, "reasoner", schema=BenGrahamSignal
    )
    prompt = render_template(
        "personas/ben_graham.jinja",
        ticker=ticker,
        as_of_date=as_of_date,
        evidence=evidence,
        market_cap=state.get("market_cap"),
        query=query,
    )
    result = cast(
        BenGrahamSignal,
        await llm.ainvoke(
            [
                SystemMessage(prompt),
                HumanMessage("Render your Graham verdict now."),
            ]
        ),
    )
    return {"persona_signals": [result.model_dump()]}


# ── Data-collection sub-agent + subgraph builder ──────────────────────────────


_MCP_TOOLS = [
    "equity_fundamental_metrics",
    "equity_fundamental_income",
    "equity_fundamental_balance",
    "equity_fundamental_cash",
    "equity_historical_market_cap",
]


async def _build_data_collection_agent(config: RunnableConfig) -> CompiledStateGraph:
    """Compiled ReAct sub-agent that fetches MCP data -> BenGrahamRawData."""
    cfg = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = cfg.get_llm_for_role("collector")
    mcp_tools = await get_tools(config, _MCP_TOOLS)

    builder = (
        MuffinAgentBuilder(primary, name="ben_graham_data_collection")
        .with_fallback_models(*fallbacks)
        .with_state_schema(BenGrahamState)
        .with_runtime_system_prompt_template(
            "personas/ben_graham_data_collection.jinja"
        )
        .with_response_format(BenGrahamRawData)
        .with_model_call_limit(run_limit=8, exit_behavior="end")
    )
    for tool in mcp_tools:
        builder = builder.with_tool(tool, run_limit=2)
    builder = builder.with_tool(execute_python, is_cacheable=False)
    return builder.build_react_agent()


async def build_ben_graham_agent(config: RunnableConfig) -> CompiledStateGraph:
    """Build the full 3-node Ben Graham subgraph."""
    data_agent = await _build_data_collection_agent(config)
    graph = StateGraph(
        BenGrahamState,
        input_schema=BenGrahamInput,
        output_schema=BenGrahamOutput,
    )
    graph.add_node(
        "collect_data",
        data_agent,
        input_schema=data_agent.input_schema,
        retry_policy=_LLM_RETRY,
    )
    graph.add_node("compute_evidence", compute_evidence_node)
    graph.add_node("render_verdict", render_verdict_node, retry_policy=_LLM_RETRY)
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "compute_evidence")
    graph.add_edge("compute_evidence", "render_verdict")
    graph.add_edge("render_verdict", END)
    return graph.compile()


