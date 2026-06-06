"""Position-sizing node — deterministic volatility × correlation budget.

Ports ai-hedge-fund's ``risk_manager.py`` to a LangGraph-native node:
no LLM, pure-Python.  Computes per-ticker dollar position limits
combining:

* **Volatility bucket** — annualised vol → base limit pct (15% / 20% /
  12.5% / ... / 10%)
* **Correlation multiplier** — avg correlation of this ticker with the
  portfolio's existing active positions → 0.70× … 1.10×

Plus deterministic mark-to-market of the current portfolio so position
limits are taken against the live NAV, not just starting cash.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ...portfolio.state import Portfolio, mark_to_market

logger = logging.getLogger(__name__)


# ── Output schemas ────────────────────────────────────────────────────────────


class PositionLimit(BaseModel):
    """Per-ticker position budget."""

    ticker: str
    limit_pct: float = Field(ge=0.0, le=1.0)
    """Combined limit as % of NAV (vol_pct × corr_multiplier)."""

    limit_dollars: float
    """``nav × limit_pct`` — the maximum dollar exposure to this ticker."""

    remaining_dollars: float
    """``limit_dollars − current_value`` — how much more can be added."""

    current_value: float
    """Current absolute exposure ``|long_value − short_value|`` at current price."""

    annualized_volatility: float | None
    avg_correlation: float | None
    """Average correlation with active portfolio positions (None if none)."""

    base_vol_limit_pct: float
    correlation_multiplier: float
    reasoning: str


# ── Pure scoring helpers ──────────────────────────────────────────────────────


def score_volatility_limit(annualized_volatility: float) -> float:
    """Map annualised vol → base position-limit percentage of NAV.

    Mirrors ai-hedge-fund's ``calculate_volatility_adjusted_limit`` exactly:
    base 20% multiplied by a vol-bucket multiplier (capped 25%-125%),
    yielding 5%-25% allocation range.

    * Annualised vol <15% → 25% allocation (multiplier 1.25)
    * Vol 15-30% → 20% → 12.5% (sliding)
    * Vol 30-50% → 15% → 5% (sliding)
    * Vol ≥50% → 10% allocation (multiplier 0.50)
    """
    base = 0.20
    if annualized_volatility < 0.15:
        m = 1.25
    elif annualized_volatility < 0.30:
        m = 1.0 - (annualized_volatility - 0.15) * 0.5
    elif annualized_volatility < 0.50:
        m = 0.75 - (annualized_volatility - 0.30) * 0.5
    else:
        m = 0.50
    m = max(0.25, min(1.25, m))
    return base * m


def score_correlation_multiplier(avg_correlation: float) -> float:
    """Map avg correlation with active positions → multiplier (0.70-1.10).

    * ≥ 0.80 → 0.70× (sharp reduction — too correlated)
    * 0.60-0.80 → 0.85×
    * 0.40-0.60 → 1.00× (neutral)
    * 0.20-0.40 → 1.05× (slight bonus)
    * < 0.20 → 1.10× (diversification bonus)
    """
    if avg_correlation >= 0.80:
        return 0.70
    if avg_correlation >= 0.60:
        return 0.85
    if avg_correlation >= 0.40:
        return 1.00
    if avg_correlation >= 0.20:
        return 1.05
    return 1.10


# ── Vol + correlation calc ────────────────────────────────────────────────────


def _annualized_volatility_from_bars(
    bars: list[dict[str, Any]], lookback: int = 60
) -> tuple[float | None, list[float]]:
    """Compute annualised vol from the trailing *lookback* daily returns.

    Returns ``(vol, daily_returns)`` — ``vol`` is ``None`` when fewer than
    two valid returns are available.
    """
    closes = [b.get("close") for b in bars if b.get("close") is not None]
    returns: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev and prev > 0:
            returns.append((closes[i] - prev) / prev)
    if len(returns) < 2:
        return None, returns
    recent = returns[-lookback:] if len(returns) > lookback else returns
    if len(recent) < 2:
        return None, returns
    daily_std = statistics.pstdev(recent)
    return daily_std * math.sqrt(252), returns


def compute_correlation_matrix(
    returns_by_ticker: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Pairwise Pearson correlation across tickers; aligned on shorter series.

    Returns ``{ticker_a: {ticker_b: corr, ...}, ...}``.  Empty dict when
    fewer than 2 tickers have ≥5 returns.
    """
    valid = {t: r for t, r in returns_by_ticker.items() if len(r) >= 5}
    if len(valid) < 2:
        return {}
    out: dict[str, dict[str, float]] = {}
    tickers = list(valid)
    for i, t1 in enumerate(tickers):
        out.setdefault(t1, {})
        for t2 in tickers[i + 1 :]:
            r1, r2 = valid[t1], valid[t2]
            n = min(len(r1), len(r2))
            if n < 5:
                continue
            a = r1[-n:]
            b = r2[-n:]
            mean_a = sum(a) / n
            mean_b = sum(b) / n
            var_a = sum((x - mean_a) ** 2 for x in a)
            var_b = sum((x - mean_b) ** 2 for x in b)
            if var_a == 0 or var_b == 0:
                continue
            cov = sum((a[k] - mean_a) * (b[k] - mean_b) for k in range(n))
            corr = cov / math.sqrt(var_a * var_b)
            out[t1].setdefault(t2, corr)
            out.setdefault(t2, {}).setdefault(t1, corr)
    return out


# ── State + node ──────────────────────────────────────────────────────────────


class PositionSizingInputState(TypedDict, total=False):
    """State keys read by ``position_sizing_node``.

    ``prices_history`` is keyed by ticker with a list of OHLCV bars
    (same shape as ``PersonaDataBundle.prices_1y``).  ``current_prices``
    is needed for mark-to-market when computing the NAV the limits are
    taken against.
    """

    tickers: list[str]
    portfolio: dict[str, Any]
    prices_history: dict[str, list[dict[str, Any]]]
    current_prices: dict[str, float]


class PositionSizingOutputState(TypedDict, total=False):
    """State keys written by ``position_sizing_node``."""

    position_limits: dict[str, dict[str, Any]]
    """Mapping ticker → ``PositionLimit.model_dump()``."""


def _fallback_limit(ticker: str, reason: str) -> PositionLimit:
    return PositionLimit(
        ticker=ticker,
        limit_pct=0.0,
        limit_dollars=0.0,
        remaining_dollars=0.0,
        current_value=0.0,
        annualized_volatility=None,
        avg_correlation=None,
        base_vol_limit_pct=0.0,
        correlation_multiplier=1.0,
        reasoning=reason,
    )


async def position_sizing_node(
    state: PositionSizingInputState, config: RunnableConfig
) -> PositionSizingOutputState:
    """Compute per-ticker position limits given current portfolio state.

    Fully deterministic — no LLM call.  Reads ``state["tickers"]``,
    ``state["portfolio"]`` (Portfolio dump), ``state["prices_history"]``,
    and ``state["current_prices"]``.  Writes ``state["position_limits"]``.
    """
    tickers = state.get("tickers") or []
    portfolio_dump = state.get("portfolio") or {}
    prices_history = state.get("prices_history") or {}
    current_prices = state.get("current_prices") or {}

    if not tickers:
        return {"position_limits": {}}

    try:
        portfolio = Portfolio.model_validate(portfolio_dump)
    except Exception:
        logger.exception("position_sizing_node: invalid portfolio payload")
        return {
            "position_limits": {
                t: _fallback_limit(t, "Invalid portfolio state").model_dump()
                for t in tickers
            }
        }

    # 1. Compute per-ticker annualised vol + returns series (for correlation)
    returns_by_ticker: dict[str, list[float]] = {}
    vols: dict[str, float] = {}
    for ticker, bars in prices_history.items():
        vol, returns = _annualized_volatility_from_bars(bars)
        if returns:
            returns_by_ticker[ticker] = returns
        if vol is not None:
            vols[ticker] = vol

    # 2. Pairwise correlation across all available tickers
    corr_matrix = compute_correlation_matrix(returns_by_ticker)

    # 3. Mark-to-market the portfolio for NAV
    portfolio_value = mark_to_market(portfolio, current_prices)
    nav = portfolio_value.nav

    # 4. Identify active positions (non-zero net long-short)
    active_positions = {
        t for t, pos in portfolio.positions.items() if abs(pos.long - pos.short) > 0
    }

    # 5. Compute per-ticker limits
    limits: dict[str, PositionLimit] = {}
    for ticker in tickers:
        price = current_prices.get(ticker)
        if price is None or price <= 0:
            limits[ticker] = _fallback_limit(ticker, "No current price available")
            continue
        vol = vols.get(ticker)
        if vol is None:
            # Default to 25% annualised vol when data is missing
            vol = 0.25
        base_pct = score_volatility_limit(vol)

        # Avg correlation with active positions (or all others if no active)
        comparators = [
            t
            for t in active_positions
            if t != ticker and t in corr_matrix.get(ticker, {})
        ]
        if not comparators:
            comparators = [t for t in corr_matrix.get(ticker, {}) if t != ticker]
        avg_corr: float | None = None
        corr_mult = 1.0
        if comparators:
            corrs = [corr_matrix[ticker][t] for t in comparators]
            avg_corr = sum(corrs) / len(corrs)
            corr_mult = score_correlation_multiplier(avg_corr)

        limit_pct = base_pct * corr_mult
        limit_dollars = max(0.0, nav * limit_pct)
        pos = portfolio.positions.get(ticker)
        current_value = abs(pos.long - pos.short) * price if pos else 0.0
        remaining = max(0.0, limit_dollars - current_value)

        reasoning_parts = [
            f"Ann vol {vol:.1%} → base {base_pct:.1%}",
            f"corr mult {corr_mult:.2f}",
            f"NAV ${nav:,.0f} × {limit_pct:.1%} = ${limit_dollars:,.0f}",
            f"current ${current_value:,.0f}",
        ]
        limits[ticker] = PositionLimit(
            ticker=ticker,
            limit_pct=limit_pct,
            limit_dollars=limit_dollars,
            remaining_dollars=remaining,
            current_value=current_value,
            annualized_volatility=vol,
            avg_correlation=avg_corr,
            base_vol_limit_pct=base_pct,
            correlation_multiplier=corr_mult,
            reasoning="; ".join(reasoning_parts),
        )

    return {"position_limits": {t: lim.model_dump() for t, lim in limits.items()}}
