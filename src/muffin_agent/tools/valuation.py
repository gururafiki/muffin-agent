"""Valuation tools — WACC, DCF (blended), multiples-based fair value, scenario NAV."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel

# ── Tool 1: compute_wacc ──────────────────────────────────────────────────────


class WACCResult(BaseModel):
    """Output schema for compute_wacc."""

    wacc: float | None
    """Weighted average cost of capital as a decimal (e.g. 0.089 for 8.9%)."""

    cost_of_equity: float | None
    """CAPM cost of equity = risk_free_rate + beta × equity_risk_premium (decimal)."""

    cost_of_debt_after_tax: float | None
    """After-tax cost of debt = cost_of_debt × (1 − tax_rate) (decimal)."""


@tool(
    parse_docstring=True,
    extras={"output_schema": WACCResult.model_json_schema()},
)
def compute_wacc(
    beta: float,
    risk_free_rate: float,
    equity_risk_premium: float,
    cost_of_debt: float,
    debt_weight: float,
    equity_weight: float,
    tax_rate: float,
) -> dict[str, float | None]:
    """Compute the Weighted Average Cost of Capital (WACC) using CAPM.

    Derives cost of equity via the Capital Asset Pricing Model and blends
    with after-tax cost of debt using market-value capital-structure weights.
    The result is used as the discount rate in DCF valuation.

    Args:
        beta: Equity beta from CAPM regression (e.g. 1.2).  Obtained from
            ``risk_assessment.beta``.
        risk_free_rate: Annualised risk-free rate as a decimal (e.g. 0.045
            for 4.5%).  Use current 10-year Treasury yield or EFFR/SOFR.
        equity_risk_premium: Market equity risk premium as a decimal (e.g.
            0.055 for 5.5%).  Use Damodaran ERP estimate or 5.5% default.
        cost_of_debt: Pre-tax cost of debt as a decimal (e.g. 0.05 for 5%).
            Estimate as latest bond yield or interest expense / average debt.
        debt_weight: Debt as a fraction of total enterprise value (0–1).
            Must sum with equity_weight to approximately 1.0 (±0.01).
        equity_weight: Equity market cap as a fraction of enterprise value
            (0–1).  Must sum with debt_weight to approximately 1.0 (±0.01).
        tax_rate: Effective corporate tax rate as a decimal (e.g. 0.21 for
            21%).  Use the effective rate from the income statement.

    Returns:
        Dict with wacc (decimal), cost_of_equity (decimal),
        cost_of_debt_after_tax (decimal).  All None if weights do not
        sum to 1.0 (±0.01) or if any input is negative.
    """
    if abs(debt_weight + equity_weight - 1.0) > 0.01:
        return WACCResult(
            wacc=None,
            cost_of_equity=None,
            cost_of_debt_after_tax=None,
        ).model_dump()
    if any(v < 0 for v in [beta, risk_free_rate, cost_of_debt, debt_weight,
                             equity_weight, tax_rate]):
        return WACCResult(
            wacc=None,
            cost_of_equity=None,
            cost_of_debt_after_tax=None,
        ).model_dump()

    ke = risk_free_rate + beta * equity_risk_premium
    kd_after_tax = cost_of_debt * (1.0 - tax_rate)
    wacc = equity_weight * ke + debt_weight * kd_after_tax

    return WACCResult(
        wacc=round(wacc, 6),
        cost_of_equity=round(ke, 6),
        cost_of_debt_after_tax=round(kd_after_tax, 6),
    ).model_dump()


# ── Tool 2: compute_dcf ───────────────────────────────────────────────────────


class DCFResult(BaseModel):
    """Output schema for compute_dcf."""

    nav_per_share: float | None
    """Blended NAV per share: average of exit-multiple and Gordon Growth methods
    when both are available; single-method result otherwise."""

    nav_exit_multiple: float | None
    """NAV per share using the exit EV/EBITDA multiple terminal value method."""

    nav_gordon_growth: float | None
    """NAV per share using the Gordon Growth (perpetuity) terminal value method."""

    pv_fcfs: float | None
    """Present value of the three explicit free-cash-flow years combined."""

    pv_terminal_exit_multiple: float | None
    """PV of terminal value under the exit-multiple method."""

    pv_terminal_gordon_growth: float | None
    """PV of terminal value under the Gordon Growth method."""

    methodology: Literal["blended", "exit_multiple", "gordon_growth"] | None
    """Which terminal value method(s) were used.

    ``blended``: both methods available and averaged.
    ``exit_multiple``: only exit-multiple available.
    ``gordon_growth``: only Gordon Growth available.
    ``None``: neither method had sufficient inputs.
    """


@tool(
    parse_docstring=True,
    extras={"output_schema": DCFResult.model_json_schema()},
)
def compute_dcf(
    fcf_year1: float,
    fcf_year2: float,
    fcf_year3: float,
    wacc: float,
    shares_outstanding: float,
    net_debt: float,
    terminal_year_ebitda: float | None = None,
    exit_ebitda_multiple: float | None = None,
    terminal_growth_rate: float | None = None,
) -> dict[str, float | None]:
    """Compute blended intrinsic value per share via a 3-year explicit DCF.

    Discounts three years of explicit free-cash-flow forecasts and a terminal
    value to derive equity NAV per share.  Two terminal-value methods are
    attempted and blended when both inputs are available:

    - **Exit-multiple method**: ``TV = terminal_year_ebitda × exit_ebitda_multiple``.
      Appropriate when peer EV/EBITDA multiples are available from
      ``discovery-screening``.
    - **Gordon Growth (perpetuity) method**:
      ``TV = fcf_year3 × (1 + g) / (wacc − g)``.
      Appropriate when a long-run nominal growth rate can be estimated
      (typically GDP growth ≈ 2–3%).

    Blending: when both methods yield a result, ``nav_per_share`` is the
    simple average, reducing single-method sensitivity.

    Args:
        fcf_year1: Free cash flow in Year+1 (reporting currency units, e.g.
            millions USD).  From ``forecast.base_case.projections[0].fcf``.
        fcf_year2: Free cash flow in Year+2 (same units).
        fcf_year3: Free cash flow in Year+3 (same units).
        wacc: Discount rate as a decimal (e.g. 0.09 for 9%).  From
            ``compute_wacc`` output.
        shares_outstanding: Diluted shares outstanding in millions.  From
            ``forecast`` or ``equity-fundamentals``.
        net_debt: Net debt = total debt − cash (same currency units as FCFs;
            negative value means the company has net cash).  From
            ``forecast.base_case.projections[2].net_debt`` (Year+3 end).
        terminal_year_ebitda: EBITDA in Year+3 (same currency units as FCFs).
            Required for the exit-multiple method.  From
            ``forecast.base_case.projections[2].ebitda``.
        exit_ebitda_multiple: EV/EBITDA multiple applied to terminal EBITDA.
            Required for the exit-multiple method.  Use peer-median EV/EBITDA
            from ``discovery-screening``.
        terminal_growth_rate: Perpetuity nominal growth rate as a decimal
            (e.g. 0.025 for 2.5%).  Required for the Gordon Growth method.
            Must be strictly less than ``wacc``.

    Returns:
        Dict with nav_per_share (blended), nav_exit_multiple,
        nav_gordon_growth, pv_fcfs, pv_terminal_exit_multiple,
        pv_terminal_gordon_growth, and methodology.  Nav fields are None
        when the corresponding method lacks sufficient inputs or when
        the WACC ≤ terminal_growth_rate guard triggers.
    """
    if wacc <= 0 or shares_outstanding <= 0:
        return DCFResult(
            nav_per_share=None, nav_exit_multiple=None,
            nav_gordon_growth=None, pv_fcfs=None,
            pv_terminal_exit_multiple=None, pv_terminal_gordon_growth=None,
            methodology=None,
        ).model_dump()

    # ── Discount explicit FCFs ────────────────────────────────────────────────
    fcfs = [fcf_year1, fcf_year2, fcf_year3]
    pv_fcfs = sum(fcf / (1.0 + wacc) ** (t + 1) for t, fcf in enumerate(fcfs))

    # ── Exit-multiple terminal value ──────────────────────────────────────────
    pv_tv_exit: float | None = None
    nav_exit: float | None = None
    if terminal_year_ebitda is not None and exit_ebitda_multiple is not None:
        tv_exit = terminal_year_ebitda * exit_ebitda_multiple
        pv_tv_exit = tv_exit / (1.0 + wacc) ** 3
        equity_value_exit = pv_fcfs + pv_tv_exit - net_debt
        nav_exit = equity_value_exit / shares_outstanding

    # ── Gordon Growth terminal value ──────────────────────────────────────────
    pv_tv_gg: float | None = None
    nav_gg: float | None = None
    if terminal_growth_rate is not None:
        if wacc > terminal_growth_rate:
            spread = wacc - terminal_growth_rate
            tv_gg = fcf_year3 * (1.0 + terminal_growth_rate) / spread
            pv_tv_gg = tv_gg / (1.0 + wacc) ** 3
            equity_value_gg = pv_fcfs + pv_tv_gg - net_debt
            nav_gg = equity_value_gg / shares_outstanding

    # ── Blend ─────────────────────────────────────────────────────────────────
    available = [v for v in [nav_exit, nav_gg] if v is not None]
    _Meth = Literal["blended", "exit_multiple", "gordon_growth"] | None
    if len(available) == 2:
        nav_blended: float | None = (nav_exit + nav_gg) / 2.0  # type: ignore[operator]
        methodology: _Meth = "blended"
    elif len(available) == 1:
        nav_blended = available[0]
        methodology = "exit_multiple" if nav_exit is not None else "gordon_growth"
    else:
        nav_blended = None
        methodology = None

    return DCFResult(
        nav_per_share=round(nav_blended, 4) if nav_blended is not None else None,
        nav_exit_multiple=round(nav_exit, 4) if nav_exit is not None else None,
        nav_gordon_growth=round(nav_gg, 4) if nav_gg is not None else None,
        pv_fcfs=round(pv_fcfs, 4),
        pv_terminal_exit_multiple=(
            round(pv_tv_exit, 4) if pv_tv_exit is not None else None
        ),
        pv_terminal_gordon_growth=(
            round(pv_tv_gg, 4) if pv_tv_gg is not None else None
        ),
        methodology=methodology,
    ).model_dump()


# ── Tool 3: compute_multiples_value ──────────────────────────────────────────


@tool(parse_docstring=True)
def compute_multiples_value(
    ntm_metric: float,
    multiple: float,
    net_debt: float,
    shares_outstanding: float,
    multiple_type: Literal["ev_ebitda", "pe", "fcf_yield"],
) -> float | None:
    """Derive a per-share fair value from a valuation multiple.

    Applies a target multiple to a next-twelve-months (NTM) financial metric
    to back out an implied per-share equity value.  Three multiple types are
    supported, selected via the multiple_type argument:

    - "ev_ebitda": Enterprise value = NTM EBITDA × multiple; equity value =
      EV − net_debt; per share = equity value / shares_outstanding.
    - "pe": Per-share value = NTM EPS × multiple (ntm_metric is already
      per share; net_debt and shares_outstanding are unused).
    - "fcf_yield": Implied price = (NTM FCF / shares) / (multiple / 100),
      where multiple is the target FCF yield percentage (e.g. 5.0 for 5%).

    Args:
        ntm_metric: The next-twelve-months metric in reporting currency.
            NTM EBITDA (millions) for "ev_ebitda"; NTM diluted EPS for "pe";
            NTM free cash flow (millions) for "fcf_yield".
        multiple: The target valuation multiple to apply.
            EV/EBITDA ratio for "ev_ebitda"; P/E ratio for "pe";
            target FCF yield percentage (e.g. 5.0) for "fcf_yield".
        net_debt: Net debt = total debt minus cash in reporting currency
            (same units as ntm_metric; negative value means net cash).
            Only used for "ev_ebitda".
        shares_outstanding: Diluted shares outstanding in millions.
            Used for "ev_ebitda" and "fcf_yield"; not used for "pe".
        multiple_type: Valuation approach — one of "ev_ebitda", "pe",
            or "fcf_yield".

    Returns:
        Implied per-share fair value in reporting currency.  Returns None if
        ntm_metric ≤ 0, multiple ≤ 0, or shares_outstanding ≤ 0 (for
        methods that require it).
    """
    if ntm_metric <= 0 or multiple <= 0:
        return None

    if multiple_type == "ev_ebitda":
        if shares_outstanding <= 0:
            return None
        ev = ntm_metric * multiple
        equity_value = ev - net_debt
        return round(equity_value / shares_outstanding, 4)

    if multiple_type == "pe":
        return round(ntm_metric * multiple, 4)

    # fcf_yield
    if shares_outstanding <= 0:
        return None
    fcf_per_share = ntm_metric / shares_outstanding
    return round(fcf_per_share / (multiple / 100.0), 4)


# ── Tool 4: compute_scenario_weighted_value ───────────────────────────────────


class WeightedValuationResult(BaseModel):
    """Output schema for compute_scenario_weighted_value."""

    probability_weighted_nav: float | None
    """Probability-weighted NAV = bull × p_bull + base × p_base + bear × p_bear."""

    upside_base_pct: float | None
    """% upside from current price to base-case NAV.

    Positive = upside; negative = downside.
    Formula: (base_nav − current_price) / current_price × 100.
    """

    upside_bull_pct: float | None
    """% upside from current price to bull-case NAV."""

    downside_bear_pct: float | None
    """% change from current price to bear-case NAV.

    Typically negative (downside).
    Formula: (bear_nav − current_price) / current_price × 100.
    """

    risk_reward_ratio: float | None
    """Absolute ratio of base upside to bear downside.

    Formula: |upside_base_pct / downside_bear_pct|.  Higher is better (more
    upside per unit of downside).  None when downside_bear_pct is zero.
    """


@tool(
    parse_docstring=True,
    extras={"output_schema": WeightedValuationResult.model_json_schema()},
)
def compute_scenario_weighted_value(
    bull_nav: float,
    base_nav: float,
    bear_nav: float,
    bull_prob: float,
    base_prob: float,
    bear_prob: float,
    current_price: float,
) -> dict[str, float | None]:
    """Compute probability-weighted NAV and upside / downside metrics.

    Blends bull, base, and bear intrinsic-value estimates by their scenario
    probabilities to derive a probability-weighted NAV per share.  Also
    computes percentage upside / downside from the current market price to
    each scenario's NAV, and a risk/reward ratio (base upside ÷ bear
    downside) for quick trade assessment.

    Args:
        bull_nav: Bull-case intrinsic value per share (reporting currency).
            From ``compute_dcf`` called with bull-case FCF projections.
        base_nav: Base-case intrinsic value per share.
        bear_nav: Bear-case intrinsic value per share.
        bull_prob: Bull-case scenario probability (0–1).  From
            ``forecast.bull_case.probability``.
        base_prob: Base-case scenario probability (0–1).
        bear_prob: Bear-case scenario probability (0–1).
        current_price: Current market price per share (reporting currency).
            From ``equity_price_quote`` via the equity-price subagent.

    Returns:
        Dict with probability_weighted_nav, upside_base_pct, upside_bull_pct,
        downside_bear_pct (typically negative), and risk_reward_ratio.
        All percentage fields are None if current_price ≤ 0.
        risk_reward_ratio is None if downside_bear_pct is zero.
    """
    prob_weighted = bull_nav * bull_prob + base_nav * base_prob + bear_nav * bear_prob

    if current_price <= 0:
        return WeightedValuationResult(
            probability_weighted_nav=round(prob_weighted, 4),
            upside_base_pct=None,
            upside_bull_pct=None,
            downside_bear_pct=None,
            risk_reward_ratio=None,
        ).model_dump()

    upside_base = (base_nav - current_price) / current_price * 100.0
    upside_bull = (bull_nav - current_price) / current_price * 100.0
    downside_bear = (bear_nav - current_price) / current_price * 100.0

    if downside_bear != 0:
        rr_ratio: float | None = round(abs(upside_base / downside_bear), 4)
    else:
        rr_ratio = None

    return WeightedValuationResult(
        probability_weighted_nav=round(prob_weighted, 4),
        upside_base_pct=round(upside_base, 4),
        upside_bull_pct=round(upside_bull, 4),
        downside_bear_pct=round(downside_bear, 4),
        risk_reward_ratio=rr_ratio,
    ).model_dump()
