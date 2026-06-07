"""Shared deterministic scoring helpers for persona node functions.

Pure-Python functions called inline by persona node bodies (e.g.
``warren_buffett_node``) BEFORE the single LLM call.  Distilled from the
13 ai-hedge-fund persona files where 7–10 of these formulas were duplicated
verbatim.  Not ``@tool``-decorated — these are not LLM-callable; they
construct the ``facts`` dict the persona passes to its LLM call.

Each "score_*" function returns a :class:`Score` with ``score``,
``max_score``, and a short human-readable ``details`` string.  Each
"compute_*" function returns the canonical primitive value (a ratio,
intrinsic value, etc.) and may return ``None`` when input data is missing.

All functions are defensive: missing/``None``/empty inputs return
``Score(0, max_score, "missing data")`` or ``None`` rather than raising.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Score:
    """Lightweight sub-score result.

    Persona evidence Pydantic models wrap one of these per sub-dimension
    via :class:`muffin_agent.agents.personas_council.schemas.ScoreDetail`.  Kept as
    a frozen dataclass (not Pydantic) since these are internal computation
    results that never cross an LLM boundary directly.
    """

    score: float
    max_score: float
    details: str


# ── Defensive accessors ───────────────────────────────────────────────────────


def _first_non_none(xs: list[float | None]) -> float | None:
    for x in xs:
        if x is not None:
            return x
    return None


def _clean(xs: list[float | None]) -> list[float]:
    """Drop ``None`` entries; preserve order."""
    return [x for x in xs if x is not None]


# ── Sub-scoring helpers (used by 3+ personas) ─────────────────────────────────


def score_roe(roe: float | None) -> Score:
    """Score return on equity (Warren, Munger, Jhunjhunwala, Fisher, Lynch).

    Thresholds: >20% → 3, >15% → 2, >10% → 1, else 0.
    """
    if roe is None:
        return Score(0, 3, "ROE not available")
    if roe > 0.20:
        return Score(3, 3, f"Excellent ROE of {roe:.1%}")
    if roe > 0.15:
        return Score(2, 3, f"Strong ROE of {roe:.1%}")
    if roe > 0.10:
        return Score(1, 3, f"Decent ROE of {roe:.1%}")
    return Score(0, 3, f"Weak ROE of {roe:.1%}")


def score_debt_to_equity(de: float | None) -> Score:
    """Score debt-to-equity ratio.

    Used by Graham, Munger, Burry, Pabrai, Fisher, Druckenmiller, Damodaran.
    Thresholds: <0.3 → 3, <0.7 → 2, <1.5 → 1, else 0.
    """
    if de is None:
        return Score(0, 3, "Debt-to-equity not available")
    if de < 0.3:
        return Score(3, 3, f"Very conservative leverage (D/E {de:.2f})")
    if de < 0.7:
        return Score(2, 3, f"Conservative leverage (D/E {de:.2f})")
    if de < 1.5:
        return Score(1, 3, f"Moderate leverage (D/E {de:.2f})")
    return Score(0, 3, f"High leverage (D/E {de:.2f})")


def score_operating_margin(om: float | None) -> Score:
    """Score operating margin.

    Used by Warren, Cathie, Munger, Lynch, Fisher, Druckenmiller, Damodaran.
    Thresholds: >20% → 2, >15% → 1, else 0.
    """
    if om is None:
        return Score(0, 2, "Operating margin not available")
    if om > 0.20:
        return Score(2, 2, f"Strong operating margin {om:.1%}")
    if om > 0.15:
        return Score(1, 2, f"Decent operating margin {om:.1%}")
    return Score(0, 2, f"Weak operating margin {om:.1%}")


def score_current_ratio(cr: float | None) -> Score:
    """Score current ratio (Graham, Pabrai, Jhunjhunwala).

    Thresholds: ≥2.0 → 2, ≥1.5 → 1, else 0.  Graham himself required ≥2.0
    for "financial strength" — the tier system here lets growthier personas
    accept lower ratios while still flagging risk.
    """
    if cr is None:
        return Score(0, 2, "Current ratio not available")
    if cr >= 2.0:
        return Score(2, 2, f"Strong liquidity (current ratio {cr:.2f})")
    if cr >= 1.5:
        return Score(1, 2, f"Adequate liquidity (current ratio {cr:.2f})")
    return Score(0, 2, f"Weak liquidity (current ratio {cr:.2f})")


def score_fcf_yield(free_cash_flow: float | None, market_cap: float | None) -> Score:
    """Score free-cash-flow yield (Burry, Pabrai, Munger, Taleb, Druckenmiller).

    Yield = FCF / market_cap.  Thresholds: >10% → 4, >5% → 3, >3% → 2,
    >0 → 1, else 0.  Burry's deep-value bar (≥15%) and Pabrai's checklist
    are stricter — those personas tighten thresholds in their own scoring.
    """
    if free_cash_flow is None or market_cap is None or market_cap <= 0:
        return Score(0, 4, "FCF or market cap not available")
    fcf_yield = free_cash_flow / market_cap
    if fcf_yield > 0.10:
        return Score(4, 4, f"Very high FCF yield {fcf_yield:.1%}")
    if fcf_yield > 0.05:
        return Score(3, 4, f"Strong FCF yield {fcf_yield:.1%}")
    if fcf_yield > 0.03:
        return Score(2, 4, f"Decent FCF yield {fcf_yield:.1%}")
    if fcf_yield > 0:
        return Score(1, 4, f"Positive but thin FCF yield {fcf_yield:.1%}")
    return Score(0, 4, f"Negative FCF yield {fcf_yield:.1%}")


def score_revenue_cagr(revenues: list[float | None]) -> Score:
    """Score revenue CAGR over the supplied series.

    Used by Cathie, Fisher, Lynch, Jhunjhunwala, Druckenmiller, Damodaran.
    Expects revenues in **chronological order** (oldest → newest).  Thresholds:
    >20% → 3, >10% → 2, >5% → 1, else 0.  Returns 0 with rationale if
    insufficient data, oldest value non-positive, or fewer than two periods.
    """
    rev = _clean(revenues)
    if len(rev) < 2 or rev[0] <= 0:
        return Score(0, 3, "Insufficient revenue history for CAGR")
    periods = len(rev) - 1
    cagr = (rev[-1] / rev[0]) ** (1 / periods) - 1
    if cagr > 0.20:
        return Score(3, 3, f"Excellent revenue CAGR {cagr:.1%} over {periods}y")
    if cagr > 0.10:
        return Score(2, 3, f"Strong revenue CAGR {cagr:.1%} over {periods}y")
    if cagr > 0.05:
        return Score(1, 3, f"Decent revenue CAGR {cagr:.1%} over {periods}y")
    return Score(0, 3, f"Weak revenue CAGR {cagr:.1%} over {periods}y")


def score_eps_cagr(eps_series: list[float | None]) -> Score:
    """Score earnings-per-share CAGR (Fisher, Lynch, Jhunjhunwala, Druckenmiller).

    Expects EPS in chronological order.  Thresholds: >20% → 3, >15% → 2,
    >10% → 1, else 0.  Returns 0 if oldest EPS is non-positive (can't
    compute meaningful growth rate from negative base).
    """
    eps = _clean(eps_series)
    if len(eps) < 2 or eps[0] <= 0:
        return Score(0, 3, "Insufficient EPS history for CAGR")
    periods = len(eps) - 1
    if eps[-1] <= 0:
        return Score(0, 3, "Latest EPS negative — growth meaningless")
    cagr = (eps[-1] / eps[0]) ** (1 / periods) - 1
    if cagr > 0.20:
        return Score(3, 3, f"Excellent EPS CAGR {cagr:.1%} over {periods}y")
    if cagr > 0.15:
        return Score(2, 3, f"Strong EPS CAGR {cagr:.1%} over {periods}y")
    if cagr > 0.10:
        return Score(1, 3, f"Decent EPS CAGR {cagr:.1%} over {periods}y")
    return Score(0, 3, f"Weak EPS CAGR {cagr:.1%} over {periods}y")


def score_insider_buy_ratio(
    insider_trades: list[dict],
) -> Score:
    """Score net insider buying activity (Munger, Lynch, Fisher, Druckenmiller).

    Expects each trade dict to expose ``transaction_shares`` (positive =
    buy, negative = sell).  Thresholds on buy / (buy + sell) count:
    >70% → 8, >40% → 6, ≤40% → 4 (insider data exists but bearish), and
    5 when no data is available (neutral default per ai-hedge-fund).
    """
    if not insider_trades:
        return Score(5, 8, "No insider trading data — neutral default")
    buys = 0
    sells = 0
    for trade in insider_trades:
        shares = trade.get("transaction_shares")
        if shares is None:
            continue
        if shares > 0:
            buys += 1
        elif shares < 0:
            sells += 1
    total = buys + sells
    if total == 0:
        return Score(5, 8, "No directional insider transactions — neutral")
    ratio = buys / total
    if ratio > 0.7:
        return Score(8, 8, f"Heavy insider buying ({buys}/{total} buys)")
    if ratio > 0.4:
        return Score(6, 8, f"Balanced insider activity ({buys}/{total} buys)")
    return Score(4, 8, f"Net insider selling ({buys}/{total} buys)")


def score_margin_stability(margins: list[float | None]) -> Score:
    """Score margin stability via coefficient of variation (Munger, Fisher, Taleb).

    CV = std / |mean|.  Thresholds: <3% → 2 (very stable), <7% → 1 (stable),
    else 0 (volatile).  At least 3 data points required.
    """
    m = _clean(margins)
    if len(m) < 3:
        return Score(0, 2, "Insufficient margin history for stability")
    mean = sum(m) / len(m)
    if mean == 0:
        return Score(0, 2, "Zero mean margin — stability undefined")
    stdev = statistics.pstdev(m)
    cv = stdev / abs(mean)
    if cv < 0.03:
        return Score(2, 2, f"Highly stable margins (CV {cv:.1%})")
    if cv < 0.07:
        return Score(1, 2, f"Stable margins (CV {cv:.1%})")
    return Score(0, 2, f"Volatile margins (CV {cv:.1%})")


# ── Methodology helpers ───────────────────────────────────────────────────────


def compute_owner_earnings(
    net_income: float | None,
    depreciation_and_amortization: float | None,
    capital_expenditure: float | None,
    maintenance_capex_ratio: float = 0.75,
) -> float | None:
    """Buffett's "owner earnings" approximation.

    Formula: ``net_income + D&A − maintenance_capex``, where
    ``maintenance_capex ≈ capex × maintenance_capex_ratio`` (0.75 default
    matches the ai-hedge-fund reference: 75% of capex assumed to be
    maintenance, the remainder growth-oriented).

    Args:
        net_income: Latest net income.
        depreciation_and_amortization: Latest D&A.
        capital_expenditure: Latest capex (positive value; the function
            negates it internally).
        maintenance_capex_ratio: Fraction of capex assumed to be
            maintenance.  Default 0.75 per the upstream agent.

    Returns:
        Owner earnings in reporting currency, or ``None`` if any input is
        missing or capex is non-positive (would imply free cash position).
    """
    if (
        net_income is None
        or depreciation_and_amortization is None
        or capital_expenditure is None
        or capital_expenditure <= 0
    ):
        return None
    maintenance_capex = capital_expenditure * maintenance_capex_ratio
    return net_income + depreciation_and_amortization - maintenance_capex


def estimate_maintenance_capex(
    capex_series: list[float | None],
    depreciation_series: list[float | None],
    revenue_series: list[float | None],
) -> float:
    """Estimate maintenance capex via Buffett's median-of-three method.

    Mirrors ai-hedge-fund's ``estimate_maintenance_capex``: takes the median
    of (1) 85% of latest capex, (2) latest depreciation, (3) average
    capex/revenue ratio over the last 5 periods × latest revenue — but only
    uses method 3 when at least 3 valid capex ratios exist; otherwise returns
    the *max* of methods 1 and 2.

    All series are **oldest → newest** (latest = ``[-1]``).  Capex values may
    be signed; they are abs()'d internally.

    Args:
        capex_series: Annual capex, oldest → newest.
        depreciation_series: Annual D&A, oldest → newest.
        revenue_series: Annual revenue, oldest → newest.

    Returns:
        Estimated annual maintenance capex (always non-negative).
    """
    latest_capex = (
        abs(capex_series[-1]) if capex_series and capex_series[-1] is not None else 0.0
    )
    latest_depreciation = (
        depreciation_series[-1]
        if depreciation_series and depreciation_series[-1] is not None
        else 0.0
    )

    # capex/revenue ratios over the last 5 periods (most recent window)
    capex_ratios: list[float] = []
    for capex, revenue in zip(
        capex_series[-5:], revenue_series[-5:], strict=False
    ):
        if capex is not None and revenue is not None and revenue > 0:
            capex_ratios.append(abs(capex) / revenue)

    method_1 = latest_capex * 0.85
    method_2 = latest_depreciation

    if len(capex_ratios) >= 3:
        avg_ratio = sum(capex_ratios) / len(capex_ratios)
        latest_revenue = (
            revenue_series[-1]
            if revenue_series and revenue_series[-1] is not None
            else 0.0
        )
        method_3 = avg_ratio * latest_revenue if latest_revenue else 0.0
        return statistics.median([method_1, method_2, method_3])
    return max(method_1, method_2)


def compute_buffett_owner_earnings(
    net_income_series: list[float | None],
    depreciation_series: list[float | None],
    capex_series: list[float | None],
    revenue_series: list[float | None],
    current_assets_series: list[float | None] | None = None,
    current_liabilities_series: list[float | None] | None = None,
) -> float | None:
    """Buffett owner earnings = NI + D&A − maintenance capex − ΔWC.

    Full ai-hedge-fund parity (``calculate_owner_earnings``): maintenance
    capex via :func:`estimate_maintenance_capex` (median of three methods),
    and a working-capital-change adjustment when current-asset / -liability
    history is available (defaults to 0 when missing).

    All series are **oldest → newest** (latest = ``[-1]``).

    Args:
        net_income_series: Annual net income, oldest → newest.
        depreciation_series: Annual D&A, oldest → newest.
        capex_series: Annual capex (signed or positive), oldest → newest.
        revenue_series: Annual revenue, oldest → newest (for maintenance capex).
        current_assets_series: Annual current assets, oldest → newest (optional).
        current_liabilities_series: Annual current liabilities, oldest → newest
            (optional).

    Returns:
        Owner earnings (may be negative — callers guard ``> 0`` before the
        DCF), or ``None`` if NI / D&A / capex for the latest period are missing.
    """
    net_income = net_income_series[-1] if net_income_series else None
    depreciation = depreciation_series[-1] if depreciation_series else None
    capex = capex_series[-1] if capex_series else None
    if net_income is None or depreciation is None or capex is None:
        return None

    maintenance_capex = estimate_maintenance_capex(
        capex_series, depreciation_series, revenue_series
    )

    # Working-capital change = ΔWC over the latest two periods (0 if unavailable)
    working_capital_change = 0.0
    if current_assets_series and current_liabilities_series:
        ca = current_assets_series
        cl = current_liabilities_series
        if (
            len(ca) >= 2
            and len(cl) >= 2
            and ca[-1] is not None
            and ca[-2] is not None
            and cl[-1] is not None
            and cl[-2] is not None
        ):
            wc_curr = ca[-1] - cl[-1]
            wc_prev = ca[-2] - cl[-2]
            working_capital_change = wc_curr - wc_prev

    return net_income + depreciation - maintenance_capex - working_capital_change


def compute_graham_number(
    eps: float | None, book_value_per_share: float | None
) -> float | None:
    """Graham's intrinsic value formula: ``√(22.5 × EPS × BVPS)``.

    Both EPS and BVPS must be positive (the formula is undefined otherwise).
    The constant 22.5 = Graham's "no more than 15× earnings × 1.5× book"
    rule of thumb.

    Args:
        eps: Earnings per share (latest annual).
        book_value_per_share: Book value per share (latest annual).

    Returns:
        Graham Number per share, or ``None`` if either input is missing
        or non-positive.
    """
    if eps is None or book_value_per_share is None:
        return None
    if eps <= 0 or book_value_per_share <= 0:
        return None
    return math.sqrt(22.5 * eps * book_value_per_share)


def compute_ncav_per_share(
    current_assets: float | None,
    total_liabilities: float | None,
    outstanding_shares: float | None,
) -> float | None:
    """Graham's Net Current Asset Value (net-net) per share.

    Formula: ``(current_assets − total_liabilities) / outstanding_shares``.
    A stock trading below NCAV is Graham's classic "cigar butt" — the
    company is worth more dead than alive.

    Args:
        current_assets: Latest current assets.
        total_liabilities: Latest total liabilities (all of them, not just current).
        outstanding_shares: Latest share count.

    Returns:
        NCAV per share (can be negative — flagged in persona logic), or
        ``None`` if any input is missing or shares ≤ 0.
    """
    if (
        current_assets is None
        or total_liabilities is None
        or outstanding_shares is None
        or outstanding_shares <= 0
    ):
        return None
    return (current_assets - total_liabilities) / outstanding_shares


def compute_peg_ratio(
    pe_ratio: float | None, growth_rate_decimal: float | None
) -> float | None:
    """Peter Lynch's GARP PEG ratio.

    Formula: ``P/E ÷ (growth_rate × 100)``.  Note ai-hedge-fund's upstream
    expresses growth as a decimal (e.g. 0.15) and the formula multiplies
    by 100 to express as a percent (e.g. 15) before dividing — so PEG = 1
    means P/E equals percent-growth.  Lynch's heuristic: PEG <1 cheap,
    1–2 fair, >2 expensive.

    Args:
        pe_ratio: Price-to-earnings ratio (latest).
        growth_rate_decimal: Earnings growth rate as a decimal (e.g. 0.15 = 15%).

    Returns:
        PEG ratio, or ``None`` if inputs missing, growth ≤ 0 (no
        meaningful PEG on declining earnings), or P/E ≤ 0.
    """
    if pe_ratio is None or growth_rate_decimal is None:
        return None
    if pe_ratio <= 0 or growth_rate_decimal <= 0:
        return None
    return pe_ratio / (growth_rate_decimal * 100)


def compute_intrinsic_value_dcf(
    base_cash_flow: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    years: int = 5,
) -> float | None:
    """Compute generic single-stage DCF (Cathie Wood / Jhunjhunwala / generic).

    Projects ``base_cash_flow`` forward ``years`` years at ``growth_rate``,
    discounts each year-end cash flow at ``discount_rate``, then adds a
    Gordon Growth terminal value ``= CF_y × (1 + g) / (r − g)`` discounted
    back ``years`` years.  Returns ``None`` when Gordon Growth is undefined
    (``discount_rate ≤ terminal_growth_rate``) or ``base_cash_flow ≤ 0``.

    Args:
        base_cash_flow: Starting cash flow (e.g. FCF, owner earnings, EPS).
            Must be positive.
        growth_rate: Annual growth rate during the explicit forecast
            period, as a decimal.
        discount_rate: Required rate of return / WACC, as a decimal.
        terminal_growth_rate: Perpetual growth rate after year ``years``,
            as a decimal.  Must be strictly less than ``discount_rate``.
        years: Length of the explicit forecast period.  Default 5.

    Returns:
        Intrinsic value (sum of discounted cash flows + discounted
        terminal value), or ``None`` for invalid inputs.
    """
    if base_cash_flow <= 0 or discount_rate <= terminal_growth_rate:
        return None
    pv = 0.0
    cf = base_cash_flow
    for year in range(1, years + 1):
        cf = cf * (1 + growth_rate)
        pv += cf / ((1 + discount_rate) ** year)
    terminal_cf = cf * (1 + terminal_growth_rate)
    terminal_value = terminal_cf / (discount_rate - terminal_growth_rate)
    pv += terminal_value / ((1 + discount_rate) ** years)
    return pv


def compute_intrinsic_value_exit_multiple(
    base_cash_flow: float,
    growth_rate: float,
    discount_rate: float,
    terminal_multiple: float,
    years: int = 5,
) -> float | None:
    """Single-stage DCF with exit-multiple terminal value (Cathie Wood).

    Projects ``base_cash_flow`` forward ``years`` years at ``growth_rate``,
    discounts each cash flow, then adds a terminal value of
    ``CF_y × terminal_multiple`` discounted back ``years`` years.
    Matches Cathie Wood's high-growth DCF (20% growth, 15% disc, 25× term).

    Args:
        base_cash_flow: Starting cash flow.  Must be positive.
        growth_rate: Annual growth rate, decimal.
        discount_rate: Required return, decimal.  Must be > 0.
        terminal_multiple: Exit multiple applied to year-``years`` cash
            flow (e.g. 25.0 for a 25× exit multiple).
        years: Explicit forecast period.

    Returns:
        Intrinsic value, or ``None`` for non-positive inputs.
    """
    if base_cash_flow <= 0 or discount_rate <= 0 or terminal_multiple <= 0:
        return None
    pv = 0.0
    cf = base_cash_flow
    for year in range(1, years + 1):
        cf = cf * (1 + growth_rate)
        pv += cf / ((1 + discount_rate) ** year)
    terminal_value = cf * terminal_multiple
    pv += terminal_value / ((1 + discount_rate) ** years)
    return pv


def compute_buffett_3stage_dcf(
    owner_earnings: float,
    growth_stage_1: float = 0.08,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    conservatism_factor: float = 0.85,
    stage_1_years: int = 5,
    stage_2_years: int = 5,
    growth_stage_2: float | None = None,
) -> float | None:
    """Warren Buffett's 3-stage DCF on owner earnings.

    Stage 1: 5 years at ``growth_stage_1`` (capped at 8%, conservative even
    for a strong moat).  Stage 2: 5 years at ``growth_stage_2`` (fading
    competitive advantage; defaults to half of stage-1 growth when not
    supplied).  Stage 3: perpetual ``terminal_growth`` (~2.5%, long-run GDP).
    Applies a ``conservatism_factor`` haircut (default 0.85, matching the
    ai-hedge-fund reference) to the final intrinsic value to enforce
    additional margin of safety on the calculation itself.

    The upstream ai-hedge-fund agent derives stage-1 growth from historical
    net-income CAGR (clamped to −5%..+15%, ×0.7 haircut, then capped at 8%)
    and sets stage-2 growth to ``min(stage_1 × 0.5, 4%)``.  Callers wanting
    exact upstream parity should compute those values and pass both
    ``growth_stage_1`` and ``growth_stage_2`` explicitly.

    Args:
        owner_earnings: Starting owner earnings (from
            :func:`compute_buffett_owner_earnings`).  Must be positive.
        growth_stage_1: Year 1–5 growth rate, decimal.  Default 8%, capped
            by the conservative Buffett bias.
        discount_rate: Required return, decimal.  Default 10%.
        terminal_growth: Perpetual growth post-stage-2, decimal.  Default 2.5%.
        conservatism_factor: Final-value haircut (0.0–1.0).  Default 0.85.
        stage_1_years: Length of stage 1.  Default 5.
        stage_2_years: Length of stage 2.  Default 5.
        growth_stage_2: Year 6–10 growth rate, decimal.  When ``None``
            (default) this is ``growth_stage_1 / 2`` — pass an explicit value
            (e.g. ``min(growth_stage_1 * 0.5, 0.04)``) for upstream parity.

    Returns:
        Intrinsic value with haircut applied, or ``None`` if owner_earnings
        is non-positive or discount_rate ≤ terminal_growth.
    """
    if owner_earnings <= 0 or discount_rate <= terminal_growth:
        return None

    if growth_stage_2 is None:
        growth_stage_2 = growth_stage_1 / 2.0
    pv = 0.0
    cf = owner_earnings

    # Stage 1: full growth
    for year in range(1, stage_1_years + 1):
        cf = cf * (1 + growth_stage_1)
        pv += cf / ((1 + discount_rate) ** year)

    # Stage 2: half growth (fading moat)
    for year in range(stage_1_years + 1, stage_1_years + stage_2_years + 1):
        cf = cf * (1 + growth_stage_2)
        pv += cf / ((1 + discount_rate) ** year)

    # Stage 3: Gordon Growth terminal value
    terminal_cf = cf * (1 + terminal_growth)
    terminal_value = terminal_cf / (discount_rate - terminal_growth)
    pv += terminal_value / ((1 + discount_rate) ** (stage_1_years + stage_2_years))

    return pv * conservatism_factor


def compute_damodaran_fcff_dcf(
    base_fcff: float,
    initial_growth: float,
    beta: float,
    risk_free_rate: float = 0.04,
    equity_risk_premium: float = 0.05,
    terminal_growth: float = 0.025,
    years: int = 10,
    terminal_basis: Literal["base_fcff", "final_cf"] = "base_fcff",
) -> tuple[float, float] | None:
    """Aswath Damodaran's 10-year FCFF DCF with CAPM cost of equity.

    Uses CAPM to derive discount rate: ``r = rf + β × ERP``.  Growth fades
    linearly from ``initial_growth`` in year 1 to ``terminal_growth`` in
    year ``years``.  Returns a tuple of (intrinsic_value, discount_rate_used)
    so the persona can cite the WACC in its reasoning.

    ``terminal_basis`` controls the Gordon-Growth terminal value:

    * ``"base_fcff"`` (default) — ``base_fcff × (1 + tg) / (r − tg)`` discounted
      back ``years`` periods.  **This matches the upstream ai-hedge-fund agent
      verbatim.**  It anchors the perpetuity on the *un-grown* base FCFF, which
      understates the terminal value relative to a textbook DCF — kept for
      exact parity with the source.
    * ``"final_cf"`` — ``final_year_cf × (1 + tg) / (r − tg)`` (textbook Gordon
      Growth on the grown year-``years`` cash flow).  Produces a higher, more
      conventional intrinsic value.

    Args:
        base_fcff: Starting free cash flow to firm.  Must be positive.
        initial_growth: Year-1 growth rate, decimal.  Capped internally
            at 12% to stay defensible.
        beta: Equity beta vs market.  Used in CAPM.
        risk_free_rate: 10y treasury yield equivalent, decimal.  Default 4%.
        equity_risk_premium: Market risk premium over risk-free, decimal.
            Default 5%.
        terminal_growth: Perpetual growth, decimal.  Default 2.5%.
        years: Explicit forecast period.  Default 10.
        terminal_basis: Terminal-value anchor; see above.  Default
            ``"base_fcff"`` for upstream parity.

    Returns:
        Tuple ``(intrinsic_value, discount_rate)``, or ``None`` if
        base_fcff ≤ 0 or the derived discount rate doesn't exceed terminal_growth.
    """
    if base_fcff <= 0:
        return None
    discount_rate = risk_free_rate + beta * equity_risk_premium
    if discount_rate <= terminal_growth:
        return None

    growth = min(initial_growth, 0.12)
    # Linear fade from initial_growth to terminal_growth over `years` periods
    growth_steps = [
        growth - (growth - terminal_growth) * (t / (years - 1)) for t in range(years)
    ]

    pv = 0.0
    cf = base_fcff
    for t, g in enumerate(growth_steps, start=1):
        cf = cf * (1 + g)
        pv += cf / ((1 + discount_rate) ** t)

    terminal_anchor = base_fcff if terminal_basis == "base_fcff" else cf
    terminal_cf = terminal_anchor * (1 + terminal_growth)
    terminal_value = terminal_cf / (discount_rate - terminal_growth)
    pv += terminal_value / ((1 + discount_rate) ** years)
    return pv, discount_rate


# ── Price-series helpers (Taleb / Druckenmiller) ──────────────────────────────


def compute_volatility_metrics(
    daily_returns: list[float],
) -> dict[str, float | None]:
    """Compute annualised vol, skew, excess kurtosis, max drawdown.

    Used by Nassim Taleb (tail-risk scoring) and Stanley Druckenmiller
    (risk-reward).  Inputs are arithmetic daily returns as decimals (e.g.
    0.01 = 1%); function returns a dict with ``annualized_volatility``,
    ``skewness``, ``excess_kurtosis``, ``max_drawdown_pct``.  All values
    are ``None`` when fewer than 4 observations are provided (skew /
    kurtosis need 4+).

    Args:
        daily_returns: Arithmetic daily return series, oldest first.

    Returns:
        Dict with the four metric fields.  ``None`` values for any metric
        the input is too short to support.
    """
    if len(daily_returns) < 4:
        return {
            "annualized_volatility": None,
            "skewness": None,
            "excess_kurtosis": None,
            "max_drawdown_pct": None,
        }
    n = len(daily_returns)
    mean = sum(daily_returns) / n
    # Population convention throughout (÷ n) so the stdev used below is on the
    # same basis as the 3rd/4th moments — Pearson moment skewness / excess
    # kurtosis are only well-defined when numerator and denominator share a
    # convention. (Previously the variance used n−1 while the moments used n,
    # mixing conventions.)
    variance = sum((r - mean) ** 2 for r in daily_returns) / n
    stdev = math.sqrt(variance)
    annual_vol = stdev * math.sqrt(252)

    # Pearson moment skewness and excess kurtosis (population convention)
    if stdev > 0:
        m3 = sum((r - mean) ** 3 for r in daily_returns) / n
        m4 = sum((r - mean) ** 4 for r in daily_returns) / n
        skewness = m3 / (stdev**3)
        excess_kurtosis = m4 / (stdev**4) - 3.0
    else:
        skewness = None
        excess_kurtosis = None

    # Max drawdown derived from compounded equity curve
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in daily_returns:
        equity *= 1 + r
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (equity - peak) / peak
            if dd < max_dd:
                max_dd = dd

    return {
        "annualized_volatility": annual_vol,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "max_drawdown_pct": max_dd * 100 if max_dd != 0 else 0.0,
    }


def compute_price_momentum(
    prices: list[float],
) -> dict[str, float | None]:
    """Period total return + recent vs early return spread (Druckenmiller).

    Args:
        prices: Closing-price series, oldest first.

    Returns:
        Dict with ``total_return_pct`` (% change from ``prices[0]`` to
        ``prices[-1]``) and ``recent_vs_early_pct`` (last-quarter return
        minus first-quarter return, both as percent — captures momentum
        acceleration / deceleration).  Values are ``None`` if fewer than
        4 data points or oldest price is non-positive.
    """
    if len(prices) < 4 or prices[0] <= 0:
        return {"total_return_pct": None, "recent_vs_early_pct": None}

    total_return = (prices[-1] - prices[0]) / prices[0]

    quarter = max(1, len(prices) // 4)
    early_segment = prices[:quarter]
    recent_segment = prices[-quarter:]
    if early_segment[0] <= 0 or recent_segment[0] <= 0:
        recent_vs_early: float | None = None
    else:
        early_return = (early_segment[-1] - early_segment[0]) / early_segment[0]
        recent_return = (recent_segment[-1] - recent_segment[0]) / recent_segment[0]
        recent_vs_early = (recent_return - early_return) * 100

    return {
        "total_return_pct": total_return * 100,
        "recent_vs_early_pct": recent_vs_early,
    }
