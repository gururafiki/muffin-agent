"""Tests for the outcomes-extraction helpers (PR 4).

The MCP-backed ``fetch_outcomes_openbb`` is not tested end-to-end here —
it's exercised via the ``OutcomesFetcher`` protocol in node and graph
tests that supply stub fetchers. These tests cover the pure-Python parsing
helpers that turn arbitrary OpenBB payloads into return numbers.
"""

import pytest

from muffin_agent.agents.trading_decision.reflection.outcomes import (
    _add_calendar_buffer,
    _extract_closes,
    _window_return,
)


@pytest.mark.unit
class TestAddCalendarBuffer:
    def test_adds_buffer_for_short_window(self):
        result = _add_calendar_buffer("2026-05-17", 5)
        # 5 trading days * 1.8 + 3 = 12 calendar days.
        assert result == "2026-05-29"

    def test_adds_buffer_for_longer_window(self):
        result = _add_calendar_buffer("2026-05-17", 20)
        # 20 * 1.8 + 3 = 39 calendar days = "2026-06-25".
        assert result == "2026-06-25"


@pytest.mark.unit
class TestExtractCloses:
    def test_plain_list_of_dicts(self):
        raw = [
            {"date": "2026-05-17", "close": 195.0},
            {"date": "2026-05-18", "close": 197.5},
            {"date": "2026-05-19", "close": 196.0},
        ]
        result = _extract_closes(raw)
        assert result == [
            ("2026-05-17", 195.0),
            ("2026-05-18", 197.5),
            ("2026-05-19", 196.0),
        ]

    def test_wrapped_in_results_key(self):
        raw = {
            "results": [
                {"date": "2026-05-17", "close": 195.0},
                {"date": "2026-05-18", "close": 198.0},
            ]
        }
        result = _extract_closes(raw)
        assert result == [("2026-05-17", 195.0), ("2026-05-18", 198.0)]

    def test_accepts_adj_close(self):
        raw = [{"date": "2026-05-17", "adj_close": 195.5}]
        assert _extract_closes(raw) == [("2026-05-17", 195.5)]

    def test_normalises_date_to_yyyy_mm_dd(self):
        raw = [{"date": "2026-05-17T00:00:00Z", "close": 195.0}]
        assert _extract_closes(raw) == [("2026-05-17", 195.0)]

    def test_drops_rows_missing_fields(self):
        raw = [
            {"date": "2026-05-17", "close": 195.0},
            {"date": "2026-05-18"},  # no close
            {"close": 197.0},  # no date
            "not a dict",
        ]
        assert _extract_closes(raw) == [("2026-05-17", 195.0)]

    def test_sorts_output_by_date(self):
        raw = [
            {"date": "2026-05-19", "close": 196.0},
            {"date": "2026-05-17", "close": 195.0},
            {"date": "2026-05-18", "close": 197.5},
        ]
        result = _extract_closes(raw)
        assert [d for d, _ in result] == ["2026-05-17", "2026-05-18", "2026-05-19"]

    def test_unrecognised_payload_returns_empty(self):
        assert _extract_closes("garbage") == []
        assert _extract_closes(None) == []


@pytest.mark.unit
class TestWindowReturn:
    def _closes(self) -> list[tuple[str, float]]:
        return [
            ("2026-05-17", 195.0),
            ("2026-05-18", 197.0),
            ("2026-05-19", 199.0),
            ("2026-05-20", 198.0),
            ("2026-05-21", 200.0),
            ("2026-05-22", 205.0),  # holding day 5
        ]

    def test_basic_5_day_return(self):
        result = _window_return(self._closes(), "2026-05-17", 5)
        assert result is not None
        return_pct, days = result
        # (205 - 195) / 195 * 100 = 5.128...
        assert return_pct == pytest.approx(5.128, rel=1e-3)
        assert days == 5

    def test_shorter_window_when_insufficient_data(self):
        # Only 3 trading days after start.
        short_closes = [
            ("2026-05-17", 195.0),
            ("2026-05-18", 197.0),
            ("2026-05-19", 199.0),
        ]
        result = _window_return(short_closes, "2026-05-17", 5)
        assert result is not None
        # Caps at the last available day (index 2).
        _, days = result
        assert days == 2

    def test_skips_dates_before_decision(self):
        with_history = [
            ("2026-05-10", 190.0),  # before decision
            ("2026-05-11", 191.0),  # before decision
            *self._closes(),
        ]
        result = _window_return(with_history, "2026-05-17", 5)
        assert result is not None
        return_pct, _ = result
        # Should still start at 195.0, not 190.0.
        assert return_pct == pytest.approx(5.128, rel=1e-3)

    def test_returns_none_when_insufficient_data_at_start(self):
        # Decision date after all available data.
        result = _window_return([("2026-05-17", 195.0)], "2026-06-01", 5)
        assert result is None

    def test_returns_none_on_zero_start_price(self):
        result = _window_return(
            [("2026-05-17", 0.0), ("2026-05-18", 100.0)], "2026-05-17", 1
        )
        assert result is None
