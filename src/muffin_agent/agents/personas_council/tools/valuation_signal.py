"""Deterministic multi-method valuation scoring (ported from ai-hedge-fund).

Aggregates four intrinsic-value estimates — a WACC-discounted multi-scenario DCF
(0.35), Buffett owner earnings (0.35), an EV/EBITDA implied value (0.20), and a
residual-income model (0.10) — into a weighted valuation gap vs market cap, then
maps the gap to a bullish/bearish/neutral signal. No LLM. Consumed by the
``valuation`` specialist subgraph.

Upstream reference: ``ai-hedge-fund/src/agents/valuation.py`` (helper formulas
preserved; FCF series here are oldest → newest, latest = ``[-1]``).

Named ``valuation_signal`` to avoid colliding with the investment-pipeline
``tools/valuation.py`` (DCF/WACC tools for the deep-agent valuation node).
"""

from __future__ import annotations

import statistics
from typing import Any, TypedDict


class ValuationResult(TypedDict):
    """Combined valuation signal."""

    signal: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0.0–1.0
    weighted_gap: float
    market_cap: float
    methods: dict[str, dict[str, Any]]


# ── Individual valuation methods (verbatim from upstream) ─────────────────────


def calculate_owner_earnings_value(
    net_income: float | None,
    depreciation: float | None,
    capex: float | None,
    working_capital_change: float | None,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """Buffett owner-earnings valuation with a 25% margin of safety."""
    if not all(
        isinstance(x, (int, float))
        for x in (net_income, depreciation, capex, working_capital_change)
    ):
        return 0.0
    owner_earnings = net_income + depreciation - capex - working_capital_change  # type: ignore[operator]
    if owner_earnings <= 0:
        return 0.0
    pv = 0.0
    for yr in range(1, num_years + 1):
        pv += owner_earnings * (1 + growth_rate) ** yr / (1 + required_return) ** yr
    terminal_growth = min(growth_rate, 0.03)
    term_val = (
        owner_earnings * (1 + growth_rate) ** num_years * (1 + terminal_growth)
    ) / (required_return - terminal_growth)
    pv += term_val / (1 + required_return) ** num_years
    return pv * (1 - margin_of_safety)


def calculate_wacc(
    market_cap: float,
    total_debt: float | None,
    cash: float | None,
    interest_coverage: float | None,
    debt_to_equity: float | None,
    beta_proxy: float = 1.0,
    risk_free_rate: float = 0.045,
    market_risk_premium: float = 0.06,
) -> float:
    """WACC from CAPM cost of equity + coverage-implied cost of debt (6%–20%)."""
    cost_of_equity = risk_free_rate + beta_proxy * market_risk_premium
    if interest_coverage and interest_coverage > 0:
        cost_of_debt = max(
            risk_free_rate + 0.01, risk_free_rate + (10 / interest_coverage)
        )
    else:
        cost_of_debt = risk_free_rate + 0.05
    net_debt = max((total_debt or 0) - (cash or 0), 0)
    total_value = market_cap + net_debt
    if total_value > 0:
        weight_equity = market_cap / total_value
        weight_debt = net_debt / total_value
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * 0.75)
    else:
        wacc = cost_of_equity
    return min(max(wacc, 0.06), 0.20)


def _fcf_volatility(fcf_history: list[float]) -> float:
    if len(fcf_history) < 3:
        return 0.5
    positive = [f for f in fcf_history if f > 0]
    if len(positive) < 2:
        return 0.8
    mean = statistics.mean(positive)
    if mean <= 0:
        return 0.8
    return min(statistics.stdev(positive) / mean, 1.0)


def _enhanced_dcf_value(
    fcf_history: list[float],
    wacc: float,
    market_cap: float,
    revenue_growth: float | None,
) -> float:
    """Multi-stage DCF (3y high growth, 4y transition, terminal).

    ``fcf_history`` is oldest → newest (latest = ``[-1]``).
    """
    if not fcf_history or fcf_history[-1] <= 0:
        return 0.0
    fcf_current = fcf_history[-1]
    fcf_avg_3yr = sum(fcf_history[-3:]) / min(3, len(fcf_history))

    high_growth = min(revenue_growth or 0.05, 0.25) if revenue_growth else 0.05
    if market_cap > 50_000_000_000:
        high_growth = min(high_growth, 0.10)
    transition_growth = (high_growth + 0.03) / 2
    terminal_growth = min(0.03, high_growth * 0.6)

    pv = 0.0
    base_fcf = max(fcf_current, fcf_avg_3yr * 0.85)
    for year in range(1, 4):
        pv += base_fcf * (1 + high_growth) ** year / (1 + wacc) ** year
    for year in range(4, 8):
        transition_rate = transition_growth * (8 - year) / 4
        fcf_proj = (
            base_fcf * (1 + high_growth) ** 3 * (1 + transition_rate) ** (year - 3)
        )
        pv += fcf_proj / (1 + wacc) ** year

    final_fcf = base_fcf * (1 + high_growth) ** 3 * (1 + transition_growth) ** 4
    if wacc <= terminal_growth:
        terminal_growth = wacc * 0.8
    terminal_value = (final_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 7
    quality_factor = max(0.7, 1 - (_fcf_volatility(fcf_history) * 0.5))
    return (pv + pv_terminal) * quality_factor


def calculate_dcf_scenarios(
    fcf_history: list[float],
    wacc: float,
    market_cap: float,
    revenue_growth: float | None,
) -> dict[str, float]:
    """Probability-weighted (0.2/0.6/0.2) bear/base/bull DCF expected value."""
    scenarios = {
        "bear": {"growth_adj": 0.5, "wacc_adj": 1.2},
        "base": {"growth_adj": 1.0, "wacc_adj": 1.0},
        "bull": {"growth_adj": 1.5, "wacc_adj": 0.9},
    }
    base_rev = revenue_growth or 0.05
    results: dict[str, float] = {}
    for name, adj in scenarios.items():
        results[name] = _enhanced_dcf_value(
            fcf_history,
            wacc * adj["wacc_adj"],
            market_cap,
            base_rev * adj["growth_adj"],
        )
    expected = results["bear"] * 0.2 + results["base"] * 0.6 + results["bull"] * 0.2
    return {
        "expected_value": expected,
        "downside": results["bear"],
        "base": results["base"],
        "upside": results["bull"],
    }


def calculate_ev_ebitda_value(
    enterprise_value: float | None,
    ev_to_ebitda_history: list[float | None] | None,
    market_cap: float | None,
) -> float:
    """Implied equity value via the median EV/EBITDA multiple."""
    ratios = [v for v in (ev_to_ebitda_history or []) if v]
    latest_ratio = ratios[-1] if ratios else None
    if not enterprise_value or not latest_ratio or latest_ratio == 0:
        return 0.0
    ebitda_now = enterprise_value / latest_ratio
    median_mult = statistics.median(ratios)
    ev_implied = median_mult * ebitda_now
    net_debt = (enterprise_value or 0) - (market_cap or 0)
    return max(ev_implied - net_debt, 0.0)


def calculate_residual_income_value(
    market_cap: float | None,
    net_income: float | None,
    price_to_book_ratio: float | None,
    book_value_growth: float = 0.03,
    cost_of_equity: float = 0.10,
    terminal_growth_rate: float = 0.03,
    num_years: int = 5,
) -> float:
    """Edwards–Bell–Ohlson residual income model with a 20% haircut."""
    if not (
        market_cap and net_income and price_to_book_ratio and price_to_book_ratio > 0
    ):
        return 0.0
    book_val = market_cap / price_to_book_ratio
    ri0 = net_income - cost_of_equity * book_val
    if ri0 <= 0:
        return 0.0
    pv_ri = 0.0
    for yr in range(1, num_years + 1):
        pv_ri += ri0 * (1 + book_value_growth) ** yr / (1 + cost_of_equity) ** yr
    term_ri = (
        ri0 * (1 + book_value_growth) ** (num_years + 1)
    ) / (cost_of_equity - terminal_growth_rate)
    pv_term = term_ri / (1 + cost_of_equity) ** num_years
    return (book_val + pv_ri + pv_term) * 0.8


# ── Aggregation ───────────────────────────────────────────────────────────────


def score_valuation_signals(
    *,
    market_cap: float | None,
    net_income: float | None,
    depreciation: float | None,
    capital_expenditure: float | None,
    working_capital_change: float | None,
    earnings_growth: float | None,
    revenue_growth: float | None,
    free_cash_flow_history: list[float | None] | None,
    total_debt: float | None,
    cash: float | None,
    interest_coverage: float | None,
    debt_to_equity: float | None,
    enterprise_value: float | None,
    ev_to_ebitda_history: list[float | None] | None,
    price_to_book_ratio: float | None,
    book_value_growth: float | None,
) -> ValuationResult:
    """Weighted multi-method valuation gap vs market cap.

    signal: weighted gap >+15% → bullish, <−15% → bearish, else neutral.
    confidence = ``min(|gap| / 0.30, 1.0)``.
    """
    fcf_history = [v for v in (free_cash_flow_history or []) if v is not None]

    wacc = calculate_wacc(
        market_cap or 0.0, total_debt, cash, interest_coverage, debt_to_equity
    )
    dcf = (
        calculate_dcf_scenarios(fcf_history, wacc, market_cap or 0.0, revenue_growth)
        if fcf_history
        else {"expected_value": 0.0, "downside": 0.0, "base": 0.0, "upside": 0.0}
    )
    owner_val = calculate_owner_earnings_value(
        net_income,
        depreciation,
        capital_expenditure,
        working_capital_change,
        growth_rate=earnings_growth or 0.05,
    )
    ev_ebitda_val = calculate_ev_ebitda_value(
        enterprise_value, ev_to_ebitda_history, market_cap
    )
    rim_val = calculate_residual_income_value(
        market_cap, net_income, price_to_book_ratio, book_value_growth or 0.03
    )

    methods: dict[str, dict[str, Any]] = {
        "dcf": {"value": dcf["expected_value"], "weight": 0.35},
        "owner_earnings": {"value": owner_val, "weight": 0.35},
        "ev_ebitda": {"value": ev_ebitda_val, "weight": 0.20},
        "residual_income": {"value": rim_val, "weight": 0.10},
    }

    if not market_cap or market_cap <= 0:
        return ValuationResult(
            signal="neutral",
            confidence=0.0,
            weighted_gap=0.0,
            market_cap=market_cap or 0.0,
            methods=methods,
        )

    total_weight = sum(m["weight"] for m in methods.values() if m["value"] > 0)
    if total_weight == 0:
        return ValuationResult(
            signal="neutral",
            confidence=0.0,
            weighted_gap=0.0,
            market_cap=market_cap,
            methods=methods,
        )

    for m in methods.values():
        m["gap"] = (m["value"] - market_cap) / market_cap if m["value"] > 0 else None

    weighted_gap = (
        sum(m["weight"] * m["gap"] for m in methods.values() if m["gap"] is not None)
        / total_weight
    )
    if weighted_gap > 0.15:
        signal = "bullish"
    elif weighted_gap < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"
    confidence = min(abs(weighted_gap) / 0.30, 1.0)
    methods["dcf"]["scenarios"] = dcf
    methods["dcf"]["wacc"] = wacc

    return ValuationResult(
        signal=signal,
        confidence=confidence,
        weighted_gap=weighted_gap,
        market_cap=market_cap,
        methods=methods,
    )
