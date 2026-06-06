"""Unit tests for persona schemas, data bundle, and registry scaffolding."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from muffin_agent.agents.personas import (
    LINE_ITEM_FIELDS,
    PERSONA_REGISTRY,
    AnalystSignal,
    InsiderTrade,
    MarketCapHistoryPoint,
    NewsArticle,
    PersonaDataBundle,
    PersonaSpec,
    PriceBar,
    ScoreDetail,
    register_persona,
)


@pytest.mark.unit
class TestAnalystSignal:
    def test_minimal_construction(self):
        s = AnalystSignal(
            agent_id="warren_buffett",
            signal="strong_buy",
            confidence=0.85,
            reasoning="Wide moat, generous owner earnings, 20% MOS.",
        )
        assert s.evidence == {}

    def test_signal_must_be_5_tier(self):
        with pytest.raises(ValidationError):
            AnalystSignal(
                agent_id="x",
                signal="bullish",  # type: ignore[arg-type]  # not a 5-tier value
                confidence=0.5,
                reasoning="r",
            )

    def test_confidence_must_be_bounded(self):
        with pytest.raises(ValidationError):
            AnalystSignal(agent_id="x", signal="buy", confidence=1.5, reasoning="r")
        with pytest.raises(ValidationError):
            AnalystSignal(agent_id="x", signal="buy", confidence=-0.1, reasoning="r")

    def test_evidence_is_freeform_by_default(self):
        s = AnalystSignal(
            agent_id="x",
            signal="hold",
            confidence=0.5,
            reasoning="r",
            evidence={"custom_field": 42, "nested": {"k": "v"}},
        )
        assert s.evidence["custom_field"] == 42


@pytest.mark.unit
class TestScoreDetail:
    def test_basic_construction(self):
        d = ScoreDetail(
            score=3.0,
            max_score=5.0,
            details="strong ROE",
            metrics={"roe": 0.18},
        )
        assert d.score == 3.0
        assert d.metrics["roe"] == 0.18

    def test_metrics_default_empty(self):
        d = ScoreDetail(score=0.0, max_score=5.0, details="missing")
        assert d.metrics == {}


@pytest.mark.unit
class TestPersonaDataBundle:
    def test_minimal_construction(self):
        b = PersonaDataBundle(ticker="AAPL", as_of_date="2025-01-31")
        assert b.financial_metrics == []
        assert b.line_items == {}
        assert b.market_cap is None
        assert b.insider_trades == []

    def test_full_construction(self):
        b = PersonaDataBundle(
            ticker="MSFT",
            as_of_date="2025-01-31",
            financial_metrics=[{"return_on_equity": 0.30}],
            line_items={"revenue": [100, 110, 120]},
            market_cap=3e12,
            market_cap_history=[
                MarketCapHistoryPoint(date="2025-01-01", market_cap=3e12)
            ],
            insider_trades=[
                InsiderTrade(transaction_shares=1000.0, insider_name="Satya Nadella")
            ],
            company_news=[
                NewsArticle(
                    date="2025-01-15",
                    title="Microsoft earnings beat",
                    sentiment="positive",
                )
            ],
            prices_1y=[
                PriceBar(
                    date="2025-01-31",
                    open=400,
                    high=405,
                    low=398,
                    close=403,
                    volume=1e7,
                )
            ],
            data_quality_notes=["only 5 years of fundamentals"],
        )
        assert b.market_cap == 3e12
        assert len(b.insider_trades) == 1
        assert b.insider_trades[0].transaction_shares == 1000.0

    def test_line_item_fields_constant(self):
        # Sanity check on the canonical line-item set
        assert isinstance(LINE_ITEM_FIELDS, tuple)
        assert "revenue" in LINE_ITEM_FIELDS
        assert "free_cash_flow" in LINE_ITEM_FIELDS
        assert "outstanding_shares" in LINE_ITEM_FIELDS
        assert len(LINE_ITEM_FIELDS) >= 28


@pytest.mark.unit
class TestPersonaRegistry:
    def teardown_method(self):
        # Remove any test entries we may have added so other tests aren't
        # affected by ordering. Use list() to avoid mutating during iteration.
        for slug in list(PERSONA_REGISTRY):
            if slug.startswith("test_"):
                del PERSONA_REGISTRY[slug]

    def test_registry_initially_empty(self):
        # Real personas haven't been imported in this test context (Phase 2
        # populates them).  May contain entries from prior tests' teardown.
        # Just verify shape.
        assert isinstance(PERSONA_REGISTRY, dict)

    def test_register_and_retrieve(self):
        async def dummy_node(state, config):
            return {"persona_signals": []}

        spec = PersonaSpec(
            slug="test_dummy",
            display_name="Test Dummy",
            investing_style="Test only",
            node=dummy_node,
            signal_schema=AnalystSignal,
        )
        register_persona(spec)
        assert PERSONA_REGISTRY["test_dummy"] is spec

    def test_duplicate_slug_raises(self):
        async def dummy_node(state, config):
            return {"persona_signals": []}

        spec = PersonaSpec(
            slug="test_dup",
            display_name="x",
            investing_style="x",
            node=dummy_node,
            signal_schema=AnalystSignal,
        )
        register_persona(spec)
        with pytest.raises(ValueError, match="already registered"):
            register_persona(spec)
