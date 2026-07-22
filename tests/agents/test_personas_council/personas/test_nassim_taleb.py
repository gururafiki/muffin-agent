"""Regression tests for nassim_taleb numeric coercion.

``NassimTalebRawData.prices_1y`` / ``.insider_trades`` are ``list[dict[str,
Any]]`` — Pydantic does not coerce their inner values, so a weak LLM can extract
``close`` / ``transaction_shares`` as strings. Comparing a string against an int
(``prev > 0``) raised ``TypeError: '>' not supported between instances of 'str'
and 'int'`` on the AMZN council run (thread 019f8476-…). ``_to_float`` guards
every dict-read site.
"""

from __future__ import annotations

import pytest

from muffin_agent.agents.personas_council.personas.nassim_taleb import (
    _daily_returns_from_bars,
    _score_taleb_skin_in_game,
    _to_float,
)


@pytest.mark.unit
class TestToFloat:
    def test_coerces_numeric_strings(self):
        assert _to_float("185.23") == 185.23
        assert _to_float(" 10 ") == 10.0
        assert _to_float(42) == 42.0
        assert _to_float(3.5) == 3.5

    def test_returns_none_for_non_numeric(self):
        assert _to_float(None) is None
        assert _to_float("") is None
        assert _to_float("n/a") is None
        assert _to_float({"x": 1}) is None

    def test_excludes_bool(self):
        # bool is an int subclass but is never a price/quantity.
        assert _to_float(True) is None
        assert _to_float(False) is None


@pytest.mark.unit
class TestDailyReturnsFromBars:
    def test_string_closes_do_not_crash(self):
        bars = [{"close": "100"}, {"close": "110"}, {"close": "99"}]
        returns = _daily_returns_from_bars(bars)
        assert returns == pytest.approx([0.1, -0.1])

    def test_mixed_and_missing_closes(self):
        bars = [{"close": 100}, {"close": None}, {"close": "abc"}, {"close": "121"}]
        # None + non-numeric drop out; remaining floats: [100.0, 121.0].
        assert _daily_returns_from_bars(bars) == pytest.approx([0.21])

    def test_empty_returns_empty(self):
        assert _daily_returns_from_bars([]) == []
        assert _daily_returns_from_bars([{"close": "not a number"}]) == []


@pytest.mark.unit
class TestSkinInGameCoercion:
    def test_string_transaction_shares_do_not_crash(self):
        state = {
            "insider_trades": [
                {"transaction_shares": "1000"},  # buy
                {"transaction_shares": "-500"},  # sell
                {"transaction_shares": None},  # ignored
            ]
        }
        evidence = _score_taleb_skin_in_game(state)  # type: ignore[arg-type]
        assert evidence.insider_buys == 1
        assert evidence.insider_sells == 1

    def test_no_insider_data(self):
        evidence = _score_taleb_skin_in_game({"insider_trades": []})  # type: ignore[arg-type]
        assert evidence.insider_buys == 0
        assert evidence.insider_sells == 0
