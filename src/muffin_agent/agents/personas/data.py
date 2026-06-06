"""Unified data bundle for the persona council.

One shared data-collection step per council run fetches everything every
persona needs and packs it into a :class:`PersonaDataBundle`.  All 13
personas then read the same bundle, avoiding 13× redundant MCP calls.

The bundle is intentionally **provider-agnostic** — it uses plain dicts /
lists for the variable-shape fields (financial_metrics, line_items,
insider_trades, company_news, prices_1y) so the underlying OpenBB MCP
shapes can evolve without breaking persona consumers.  Each persona
defensively reads only the fields it needs, falling back to ``None``
when a field is missing.

The complete data-union across 13 personas (derived from the deep audit
in the porting plan):

* **Financial metrics** (10 periods, ttm + annual):
  ``return_on_equity``, ``return_on_invested_capital``, ``debt_to_equity``,
  ``current_ratio``, ``net_margin``, ``operating_margin``, ``gross_margin``,
  ``beta``, ``interest_coverage``, ``price_to_earnings_ratio``,
  ``revenue_growth``, ``earnings_growth``, ``book_value_growth``,
  ``market_cap``, ``ev_to_ebit``, ``free_cash_flow_yield``,
  ``free_cash_flow_per_share``, ``earnings_per_share``,
  ``price_to_book_ratio``, ``price_to_sales_ratio``,
  ``dividend_yield``, ``payout_ratio``.

* **Line items** (annual, 5–10 periods): see ``LINE_ITEM_FIELDS`` constant.

* **Market cap** (latest + 5y history) — Damodaran, Cathie growth scoring.

* **Insider trades** (1y lookback) — Munger, Burry, Lynch, Fisher,
  Druckenmiller, Taleb.

* **Company news** (1y lookback, sentiment-tagged when provider supports it) —
  Munger, Burry, Lynch, Fisher, Druckenmiller, Taleb.

* **Daily OHLCV** (1y) — Taleb (vol / skew / kurtosis / drawdown),
  Druckenmiller (momentum).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Canonical line-item field names ───────────────────────────────────────────

LINE_ITEM_FIELDS: tuple[str, ...] = (
    # Income statement
    "revenue",
    "gross_profit",
    "gross_margin",
    "operating_income",
    "operating_margin",
    "operating_expense",
    "ebit",
    "ebitda",
    "net_income",
    "earnings_per_share",
    "interest_expense",
    # Cash flow statement
    "free_cash_flow",
    "capital_expenditure",
    "depreciation_and_amortization",
    "dividends_and_other_cash_distributions",
    "issuance_or_purchase_of_equity_shares",
    # Balance sheet
    "total_assets",
    "total_liabilities",
    "current_assets",
    "current_liabilities",
    "shareholders_equity",
    "book_value_per_share",
    "cash_and_equivalents",
    "total_debt",
    "outstanding_shares",
    # Other
    "research_and_development",
    "goodwill_and_intangible_assets",
    "return_on_invested_capital",
)
"""Canonical line-item field names used across all 13 personas.

The shared data-collection deep agent must populate these in
``PersonaDataBundle.line_items`` keyed by field name, with values as
chronologically-ordered lists (oldest → newest) of ``float | None``.
Missing line items map to an absent key OR an explicit empty list — both
are tolerated by the scoring helpers.
"""


# ── Bundle ─────────────────────────────────────────────────────────────────────


class PriceBar(BaseModel):
    """Single OHLCV bar.  Used in ``PersonaDataBundle.prices_1y``."""

    date: str  # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: float


class InsiderTrade(BaseModel):
    """One insider trade record.  Provider-shaped fields normalised to muffin's.

    Note ``transaction_shares`` carries sign: positive = buy, negative = sell.
    Helpers like :func:`score_insider_buy_ratio` read this directly.
    """

    filing_date: str | None = None
    transaction_date: str | None = None
    transaction_shares: float | None = None
    """Signed share count: positive buy, negative sell."""

    transaction_value: float | None = None
    """Dollar value of the transaction (may be missing on some filings)."""

    insider_name: str | None = None
    insider_title: str | None = None


class NewsArticle(BaseModel):
    """One news article record.  ``sentiment`` is provider-scored when available.

    The ``sentiment_analysis`` specialist (Phase 3.2) consumes this directly;
    personas that key on sentiment (Burry contrarian, Lynch, Druckenmiller)
    read it inline.
    """

    date: str
    title: str
    sentiment: str | None = None
    """Provider-supplied sentiment tag — ``"positive"`` / ``"negative"`` /
    ``"neutral"`` / ``None``.  When ``None``, the news_sentiment specialist
    (Phase 3.3, deferred) may LLM-score it."""

    source: str | None = None
    url: str | None = None
    impact_score: float | None = None


class MarketCapHistoryPoint(BaseModel):
    """One historical market-cap snapshot."""

    date: str  # YYYY-MM-DD
    market_cap: float


class PersonaDataBundle(BaseModel):
    """Complete data needed by ALL 13 personas. Fetched once per council run.

    Persona node functions read fields they care about and pass them
    through :mod:`muffin_agent.tools.scoring_helpers` to compute facts
    BEFORE the single LLM call.  Designed to be cacheable in muffin's
    ``ToolResultCacheMiddleware`` so repeated council runs for the same
    ticker / date don't re-fetch.
    """

    ticker: str
    """Equity ticker symbol analysed (preserved exactly as supplied — e.g.
    ``AAPL`` vs ``BNS.TO``).  Personas use this in their LLM prompt to
    keep the verdict ticker-tagged."""

    as_of_date: str
    """ISO-8601 date string the bundle is anchored to (YYYY-MM-DD).
    Drives "latest" interpretation downstream — i.e. ``financial_metrics[0]``
    is the most recent period as of this date."""

    financial_metrics: list[dict[str, Any]] = Field(default_factory=list)
    """List of period-keyed metric dicts, **newest first**.  Each dict
    exposes ratios / growth rates / yields needed by the personas.
    Provider-shaped — see module docstring for the full union of fields
    consumed across the 13 personas."""

    line_items: dict[str, list[float | None]] = Field(default_factory=dict)
    """Keyed by line item name (one of :data:`LINE_ITEM_FIELDS`); each
    value is a list of period values **chronologically ordered**
    (oldest → newest) to match what the scoring helpers expect.

    Personas that want "latest" can read ``line_items[name][-1]``; ones
    doing CAGR / consistency analysis iterate the full list."""

    market_cap: float | None = None
    """Latest market capitalisation as of ``as_of_date``."""

    market_cap_history: list[MarketCapHistoryPoint] = Field(default_factory=list)
    """Historical market-cap snapshots — typically monthly over 5 years.
    Used for Damodaran's relative-valuation check and for tracking
    long-run equity returns."""

    insider_trades: list[InsiderTrade] = Field(default_factory=list)
    """Insider trade records, 1-year lookback by convention.  Empty
    list when the issuer has no insider filings or the data is
    unavailable."""

    company_news: list[NewsArticle] = Field(default_factory=list)
    """Recent news articles, 1-year lookback.  Sentiment is provider-
    tagged when available; otherwise ``None``."""

    prices_1y: list[PriceBar] = Field(default_factory=list)
    """Daily OHLCV bars over the trailing year (or whatever window the
    upstream provider supplies).  Required by Taleb (vol / skew /
    kurtosis / drawdown) and Druckenmiller (price momentum)."""

    benchmark_prices_1y: list[PriceBar] = Field(default_factory=list)
    """Daily OHLCV bars for the comparison benchmark (e.g. SPY) over the
    same window.  Optional — only used when a persona / specialist needs
    relative returns.  Populated by the data-collection step when the
    bundle is destined for the council; left empty for standalone
    persona invocations that don't need benchmarking."""

    data_quality_notes: list[str] = Field(default_factory=list)
    """Free-form notes from the data-collection step about gaps or
    inconsistencies (e.g. "no insider trades for ticker BNS.TO",
    "only 3 years of annual financials available").  Personas reference
    this in their reasoning when their confidence is capped by data
    quality."""
