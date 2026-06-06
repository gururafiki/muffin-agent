"""Sentiment aggregation tools — insider trades + company news.

Pure-Python ports of ai-hedge-fund's ``sentiment.py`` aggregation logic.
**Fully deterministic** — no LLM is called.  Used by:

* The ``sentiment_analysis_node`` specialist (Phase 3.2) — emits one
  :class:`muffin_agent.agents.personas.AnalystSignal` summarising news +
  insider activity.
* Persona node functions that need sentiment as one of their precomputed
  facts (Burry's contrarian filter, Lynch's GARP sanity check,
  Druckenmiller's macro overlay, Fisher's qualitative reads).

Output vocabulary stays 3-tier (``bullish``/``bearish``/``neutral``)
because that's the natural shape of individual sentiment signals; the
specialist node maps to the 5-tier :class:`InvestmentSignal` when
building its :class:`AnalystSignal`.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

TacticalSignal = Literal["bullish", "bearish", "neutral"]


# ── Output schemas ────────────────────────────────────────────────────────────


class InsiderSentiment(TypedDict):
    """Aggregated insider-trading signal."""

    signal: TacticalSignal
    confidence: float  # 0.0–1.0
    total_trades: int
    bullish_trades: int
    bearish_trades: int
    net_share_change: float
    """Sum of all signed ``transaction_shares`` values.  Captures
    direction AND magnitude — useful for personas that want share-count
    momentum (Munger, Burry) rather than just trade count."""


class NewsSentiment(TypedDict):
    """Aggregated news-sentiment signal."""

    signal: TacticalSignal
    confidence: float  # 0.0–1.0
    total_articles: int
    bullish_articles: int
    bearish_articles: int
    neutral_articles: int


class CombinedSentiment(TypedDict):
    """Combined sentiment across insider + news, weighted."""

    signal: TacticalSignal
    confidence: float  # 0.0–1.0
    weighted_bullish: float
    weighted_bearish: float
    insider_weight: float
    news_weight: float
    insider: InsiderSentiment
    news: NewsSentiment


# ── Aggregation helpers ───────────────────────────────────────────────────────


def aggregate_insider_trades(insider_trades: list[dict[str, Any]]) -> InsiderSentiment:
    """Aggregate insider trades into a directional signal.

    Mirrors ai-hedge-fund's logic: ``transaction_shares < 0`` is bearish
    (a sale), ``>= 0`` is bullish.  (Zero-share trades are rare and
    are bucketed as bullish for parity with the upstream
    ``np.where(...)`` call.)

    Args:
        insider_trades: List of trade dicts each exposing a
            ``transaction_shares`` field (signed: positive buy /
            negative sell).  Trades missing this field are skipped.

    Returns:
        Dict with ``signal``, ``confidence`` (0.0–1.0), counts, and
        ``net_share_change`` (signed sum).  Empty input returns
        neutral / 0.0 confidence.
    """
    bullish = 0
    bearish = 0
    net_change = 0.0
    for trade in insider_trades:
        shares = trade.get("transaction_shares")
        if shares is None:
            continue
        net_change += float(shares)
        if shares < 0:
            bearish += 1
        else:
            bullish += 1

    total = bullish + bearish
    if total == 0:
        return InsiderSentiment(
            signal="neutral",
            confidence=0.0,
            total_trades=0,
            bullish_trades=0,
            bearish_trades=0,
            net_share_change=0.0,
        )

    if bullish > bearish:
        signal: TacticalSignal = "bullish"
        majority = bullish
    elif bearish > bullish:
        signal = "bearish"
        majority = bearish
    else:
        signal = "neutral"
        majority = max(bullish, bearish)
    confidence = majority / total

    return InsiderSentiment(
        signal=signal,
        confidence=confidence,
        total_trades=total,
        bullish_trades=bullish,
        bearish_trades=bearish,
        net_share_change=net_change,
    )


def aggregate_news_sentiment(articles: list[dict[str, Any]]) -> NewsSentiment:
    """Aggregate news articles' provider-supplied sentiment field.

    Reads the ``sentiment`` field on each article (``"positive"`` /
    ``"negative"`` / ``"neutral"`` / ``None``).  Articles without a
    sentiment tag are counted as neutral but don't contribute to the
    majority.

    Args:
        articles: List of article dicts each exposing an optional
            ``sentiment`` field.

    Returns:
        Dict with ``signal``, ``confidence`` (0.0–1.0), and per-bucket
        counts.  Empty input returns neutral / 0.0 confidence.
    """
    bullish = 0
    bearish = 0
    neutral = 0
    for article in articles:
        sent = (article.get("sentiment") or "").lower()
        if sent == "positive":
            bullish += 1
        elif sent == "negative":
            bearish += 1
        else:
            neutral += 1

    total_directional = bullish + bearish
    total = bullish + bearish + neutral

    if total == 0:
        return NewsSentiment(
            signal="neutral",
            confidence=0.0,
            total_articles=0,
            bullish_articles=0,
            bearish_articles=0,
            neutral_articles=0,
        )

    if total_directional == 0:
        signal: TacticalSignal = "neutral"
        confidence = 0.0
    elif bullish > bearish:
        signal = "bullish"
        confidence = bullish / total
    elif bearish > bullish:
        signal = "bearish"
        confidence = bearish / total
    else:
        signal = "neutral"
        confidence = max(bullish, bearish) / total

    return NewsSentiment(
        signal=signal,
        confidence=confidence,
        total_articles=total,
        bullish_articles=bullish,
        bearish_articles=bearish,
        neutral_articles=neutral,
    )


# ── Combined sentiment ────────────────────────────────────────────────────────


def combine_sentiment_signals(
    insider_trades: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    insider_weight: float = 0.3,
    news_weight: float = 0.7,
) -> CombinedSentiment:
    """Combine insider + news sentiment via weighted signal counts.

    Mirrors ai-hedge-fund's weighting (30% insider / 70% news).  The
    confidence is the share of the winning side's weighted count over
    the total weighted-signal count — bounded 0.0–1.0.

    Args:
        insider_trades: Raw insider trade dicts (signed
            ``transaction_shares``).
        articles: Raw news article dicts (with optional ``sentiment``).
        insider_weight: Weight for insider-trade signals.  Default 0.3.
        news_weight: Weight for news-sentiment signals.  Default 0.7.

    Returns:
        :class:`CombinedSentiment` carrying both the combined verdict
        and the per-source breakdowns for traceability.
    """
    insider = aggregate_insider_trades(insider_trades)
    news = aggregate_news_sentiment(articles)

    weighted_bullish = (
        insider["bullish_trades"] * insider_weight
        + news["bullish_articles"] * news_weight
    )
    weighted_bearish = (
        insider["bearish_trades"] * insider_weight
        + news["bearish_articles"] * news_weight
    )
    total_directional_weight = (
        insider["bullish_trades"] + insider["bearish_trades"]
    ) * insider_weight + (
        news["bullish_articles"] + news["bearish_articles"]
    ) * news_weight

    if total_directional_weight == 0:
        signal: TacticalSignal = "neutral"
        confidence = 0.0
    elif weighted_bullish > weighted_bearish:
        signal = "bullish"
        confidence = weighted_bullish / total_directional_weight
    elif weighted_bearish > weighted_bullish:
        signal = "bearish"
        confidence = weighted_bearish / total_directional_weight
    else:
        signal = "neutral"
        confidence = 0.5

    return CombinedSentiment(
        signal=signal,
        confidence=min(1.0, confidence),
        weighted_bullish=round(weighted_bullish, 2),
        weighted_bearish=round(weighted_bearish, 2),
        insider_weight=insider_weight,
        news_weight=news_weight,
        insider=insider,
        news=news,
    )
