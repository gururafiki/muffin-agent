"""Tests for the ``muffin decide`` CLI helper functions (PR 5).

These cover the synchronous helpers extracted from the Typer command so
the input-shape logic can be exercised without running the full
trading-decision pipeline. End-to-end CLI invocation is intentionally
not tested here — the underlying graph already has e2e coverage in
``tests/agents/test_trading_decision/``.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest
import typer

from muffin_agent.agents.trading_decision.schemas import AnalysisContext
from muffin_cli.main import _build_decide_context, _load_analysis_json


def _state(**overrides) -> dict:
    base = {
        "ticker": "AAPL",
        "query": "long-term hold",
        "market_regime": {"regime_label": "expansion"},
        "valuation": {"valuation_signal": "cheap"},
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestLoadAnalysisJson:
    def test_loads_dict_from_file(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text(json.dumps(_state()), encoding="utf-8")
        loaded = _load_analysis_json(str(target))
        assert loaded["ticker"] == "AAPL"
        assert loaded["market_regime"] == {"regime_label": "expansion"}

    def test_loads_dict_from_stdin(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(_state())))
        loaded = _load_analysis_json("-")
        assert loaded["ticker"] == "AAPL"

    def test_rejects_non_dict_json(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text(json.dumps(["a", "list"]), encoding="utf-8")
        with pytest.raises(typer.BadParameter, match="JSON object"):
            _load_analysis_json(str(target))

    def test_rejects_invalid_json(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            _load_analysis_json(str(target))

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _load_analysis_json(str(tmp_path / "missing.json"))


@pytest.mark.unit
class TestBuildDecideContext:
    def test_narrative_only_mode(self):
        ctx = _build_decide_context(
            "AAPL", narrative="Apple notes.", analysis_json=None, query=None
        )
        assert isinstance(ctx, AnalysisContext)
        assert ctx.ticker == "AAPL"
        assert ctx.narrative == "Apple notes."
        assert (
            ctx.market_regime is None
        )  # narrative mode populates no structured fields

    def test_analysis_json_only_mode(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text(json.dumps(_state()), encoding="utf-8")
        ctx = _build_decide_context(
            "AAPL",
            narrative=None,
            analysis_json=str(target),
            query="overridden query",
        )
        assert ctx.ticker == "AAPL"
        assert ctx.market_regime == {"regime_label": "expansion"}
        assert ctx.valuation == {"valuation_signal": "cheap"}
        # CLI query overrides state["query"].
        assert ctx.query == "overridden query"
        assert ctx.narrative is None

    def test_combined_analysis_and_narrative(self, tmp_path: Path):
        """Both flags layer structured fields and narrative into one context."""
        target = tmp_path / "state.json"
        target.write_text(json.dumps(_state()), encoding="utf-8")
        ctx = _build_decide_context(
            "AAPL",
            narrative="Add'l note: management transition imminent.",
            analysis_json=str(target),
            query=None,
        )
        assert ctx.market_regime == {"regime_label": "expansion"}
        assert ctx.narrative == "Add'l note: management transition imminent."
        # When CLI query is None, the state's query is preserved.
        assert ctx.query == "long-term hold"

    def test_neither_flag_raises_bad_parameter(self):
        with pytest.raises(typer.BadParameter, match="--narrative or --analysis-json"):
            _build_decide_context(
                "AAPL", narrative=None, analysis_json=None, query=None
            )

    def test_cli_ticker_takes_precedence_over_state(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text(json.dumps(_state(ticker="WRONG")), encoding="utf-8")
        ctx = _build_decide_context(
            "AAPL", narrative=None, analysis_json=str(target), query=None
        )
        assert ctx.ticker == "AAPL"

    def test_query_none_preserves_state_query(self, tmp_path: Path):
        target = tmp_path / "state.json"
        target.write_text(json.dumps(_state(query="from state")), encoding="utf-8")
        ctx = _build_decide_context(
            "AAPL", narrative=None, analysis_json=str(target), query=None
        )
        assert ctx.query == "from state"
