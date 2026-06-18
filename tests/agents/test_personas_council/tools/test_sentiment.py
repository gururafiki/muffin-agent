"""Unit tests for sentiment aggregation tools."""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.tools.sentiment import (
    aggregate_insider_trades,
    aggregate_news_sentiment,
    combine_sentiment_signals,
)


@pytest.mark.unit
class TestAggregateInsiderTrades:
    def test_all_buying(self):
        trades = [{"transaction_shares": 100} for _ in range(5)]
        result = aggregate_insider_trades(trades)
        assert result["signal"] == "bullish"
        assert result["bullish_trades"] == 5
        assert result["bearish_trades"] == 0
        assert result["confidence"] == 1.0
        assert result["net_share_change"] == 500

    def test_all_selling(self):
        trades = [{"transaction_shares": -200} for _ in range(3)]
        result = aggregate_insider_trades(trades)
        assert result["signal"] == "bearish"
        assert result["bearish_trades"] == 3
        assert result["net_share_change"] == -600

    def test_majority_buying(self):
        trades = [
            {"transaction_shares": 100},
            {"transaction_shares": 100},
            {"transaction_shares": 100},
            {"transaction_shares": -50},
        ]
        result = aggregate_insider_trades(trades)
        assert result["signal"] == "bullish"
        assert result["confidence"] == pytest.approx(0.75)

    def test_empty(self):
        result = aggregate_insider_trades([])
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0

    def test_missing_transaction_shares_skipped(self):
        trades = [
            {"transaction_shares": 100},
            {"other_field": "ignored"},  # no transaction_shares
            {"transaction_shares": -50},
        ]
        result = aggregate_insider_trades(trades)
        assert result["total_trades"] == 2
        assert result["bullish_trades"] == 1
        assert result["bearish_trades"] == 1


@pytest.mark.unit
class TestAggregateNewsSentiment:
    def test_positive_majority(self):
        articles = [{"sentiment": "positive"} for _ in range(3)] + [
            {"sentiment": "negative"}
        ]
        result = aggregate_news_sentiment(articles)
        assert result["signal"] == "bullish"
        assert result["bullish_articles"] == 3
        assert result["bearish_articles"] == 1
        assert result["confidence"] == pytest.approx(0.75)

    def test_negative_majority(self):
        articles = [{"sentiment": "negative"} for _ in range(4)] + [
            {"sentiment": "positive"}
        ]
        result = aggregate_news_sentiment(articles)
        assert result["signal"] == "bearish"

    def test_all_neutral_returns_neutral(self):
        articles = [{"sentiment": "neutral"} for _ in range(5)]
        result = aggregate_news_sentiment(articles)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0
        assert result["neutral_articles"] == 5

    def test_missing_sentiment_counted_as_neutral(self):
        articles = [
            {"sentiment": "positive"},
            {"other_field": "data"},  # no sentiment
            {"sentiment": "negative"},
        ]
        result = aggregate_news_sentiment(articles)
        assert result["total_articles"] == 3
        assert result["neutral_articles"] == 1

    def test_empty(self):
        result = aggregate_news_sentiment([])
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0


@pytest.mark.unit
class TestCombineSentimentSignals:
    def test_both_bullish(self):
        insider = [{"transaction_shares": 100} for _ in range(3)]
        news = [{"sentiment": "positive"} for _ in range(5)]
        result = combine_sentiment_signals(insider, news)
        assert result["signal"] == "bullish"
        assert result["weighted_bullish"] > result["weighted_bearish"]
        # No bearish entries → confidence should hit ceiling
        assert result["confidence"] == pytest.approx(1.0, abs=1e-9)

    def test_news_dominates_due_to_weighting(self):
        # 3 insider buys (weight 0.3 → 0.9) vs 5 news negatives (weight 0.7 → 3.5)
        insider = [{"transaction_shares": 100} for _ in range(3)]
        news = [{"sentiment": "negative"} for _ in range(5)]
        result = combine_sentiment_signals(insider, news)
        assert result["signal"] == "bearish"

    def test_custom_weights(self):
        # Flip the weighting so insider dominates
        insider = [{"transaction_shares": 100} for _ in range(3)]
        news = [{"sentiment": "negative"} for _ in range(5)]
        result = combine_sentiment_signals(
            insider, news, insider_weight=0.9, news_weight=0.1
        )
        assert result["signal"] == "bullish"

    def test_empty_both(self):
        result = combine_sentiment_signals([], [])
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.0

    def test_carries_breakdowns(self):
        insider = [{"transaction_shares": 100}]
        news = [{"sentiment": "positive"}]
        result = combine_sentiment_signals(insider, news)
        assert result["insider"]["total_trades"] == 1
        assert result["news"]["total_articles"] == 1
