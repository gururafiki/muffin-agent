"""Unit tests for ``trading_decision.tools.get_indicators``."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from muffin_agent.agents.trading_decision.tools import (
    _SUPPORTED_INDICATORS,
    _extract_results_rows,
    _normalise_for_stockstats,
    get_indicators,
)

_TOOLS_MODULE = "muffin_agent.agents.trading_decision.tools"


def _synthetic_ohlcv_rows(n: int = 60, start: str = "2025-01-01") -> list[dict]:
    """Build a ``n``-day OHLCV series with a smooth uptrend."""
    base = pd.Timestamp(start)
    return [
        {
            "date": (base + pd.Timedelta(days=i)).date().isoformat(),
            "open": 100 + i * 0.5,
            "high": 101 + i * 0.5,
            "low": 99 + i * 0.5,
            "close": 100.5 + i * 0.5,
            "volume": 1000 + i * 10,
        }
        for i in range(n)
    ]


# ── Pure-helper tests ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestExtractResultsRows:
    def test_extract_from_dict_with_results(self):
        payload = {"results": [{"date": "2025-01-01"}, {"date": "2025-01-02"}]}
        rows = _extract_results_rows(payload)
        assert len(rows) == 2

    def test_extract_from_json_string(self):
        payload = '{"results": [{"date": "2025-01-01"}]}'
        rows = _extract_results_rows(payload)
        assert rows == [{"date": "2025-01-01"}]

    def test_extract_from_bare_list(self):
        payload = [{"date": "2025-01-01"}, "ignored", {"date": "2025-01-02"}]
        rows = _extract_results_rows(payload)
        assert len(rows) == 2

    def test_extract_returns_empty_for_unknown_shape(self):
        assert _extract_results_rows({"unexpected": "shape"}) == []
        assert _extract_results_rows(None) == []

    def test_extract_raises_on_invalid_json_string(self):
        from langchain_core.tools import ToolException

        with pytest.raises(ToolException):
            _extract_results_rows("not valid json {{{")


@pytest.mark.unit
class TestNormaliseForStockstats:
    def test_normalises_columns_and_dtypes(self):
        rows = [
            {
                "Date": "2025-01-02",
                "Open": 1,
                "High": 2,
                "Low": 0.5,
                "Close": 1.5,
                "Volume": 10,
            },
            {
                "Date": "2025-01-01",
                "Open": 1,
                "High": 2,
                "Low": 0.5,
                "Close": 1.0,
                "Volume": 5,
            },
        ]
        out = _normalise_for_stockstats(pd.DataFrame(rows))
        assert list(out.columns) == [
            "date", "open", "high", "low", "close", "volume",
        ]
        assert out["date"].is_monotonic_increasing  # sorted asc

    def test_drops_rows_with_no_close(self):
        rows = [
            {
                "date": "2025-01-01",
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.0,
                "volume": 10,
            },
            {
                "date": "2025-01-02",
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": None,
                "volume": 10,
            },
        ]
        out = _normalise_for_stockstats(pd.DataFrame(rows))
        assert len(out) == 1

    def test_raises_when_date_column_missing(self):
        from langchain_core.tools import ToolException

        df = pd.DataFrame([{"close": 1.0}])
        with pytest.raises(ToolException, match="missing 'date' column"):
            _normalise_for_stockstats(df)


# ── End-to-end tool tests ─────────────────────────────────────────────────


@pytest.mark.unit
class TestGetIndicators:
    @pytest.mark.asyncio
    async def test_returns_helpful_message_for_unsupported_indicator(self):
        result = await get_indicators.ainvoke(
            {
                "ticker": "AAPL",
                "indicator": "exotic_unknown_metric",
                "curr_date": "2025-12-31",
            }
        )
        assert "Unsupported indicator" in result
        # Mentions the supported list so the agent self-corrects.
        for ind in _SUPPORTED_INDICATORS:
            assert ind in result

    @pytest.mark.asyncio
    async def test_pulls_ohlcv_and_returns_markdown_table(self):
        fake_tool = AsyncMock()
        fake_tool.ainvoke.return_value = {"results": _synthetic_ohlcv_rows(60)}

        with patch(
            f"{_TOOLS_MODULE}.get_tools",
            AsyncMock(return_value=[fake_tool]),
        ):
            md = await get_indicators.ainvoke(
                {
                    "ticker": "AAPL",
                    "indicator": "rsi",
                    "curr_date": "2025-03-01",
                    "look_back_days": 10,
                }
            )

        # Markdown table header includes the indicator column. (to_markdown
        # right-aligns numeric columns so the cell width varies; just check
        # for the column name with bordering pipes.)
        assert "| rsi |" in md.replace("  ", " ").replace("  ", " ")
        # And no error markers.
        assert "Unsupported" not in md
        assert "No OHLCV" not in md

    @pytest.mark.asyncio
    async def test_returns_no_data_message_when_openbb_empty(self):
        fake_tool = AsyncMock()
        fake_tool.ainvoke.return_value = {"results": []}

        with patch(
            f"{_TOOLS_MODULE}.get_tools",
            AsyncMock(return_value=[fake_tool]),
        ):
            result = await get_indicators.ainvoke(
                {
                    "ticker": "AAPL",
                    "indicator": "rsi",
                    "curr_date": "2025-03-01",
                }
            )
        assert "No OHLCV" in result

    @pytest.mark.asyncio
    async def test_raises_when_openbb_tool_unavailable(self):
        from langchain_core.tools import ToolException

        with patch(
            f"{_TOOLS_MODULE}.get_tools",
            AsyncMock(return_value=[]),
        ), pytest.raises(ToolException, match="equity_price_historical not available"):
            await get_indicators.ainvoke(
                {"ticker": "AAPL", "indicator": "rsi", "curr_date": "2025-03-01"}
            )
