"""Unit tests for sector tools (relative performance, peer dispersion)."""

import pytest

from muffin_agent.tools.sector import (
    compute_peer_dispersion,
    compute_sector_relative_performance,
)


@pytest.mark.unit
class TestSectorRelativePerformanceTool:
    def test_basic(self):
        result = compute_sector_relative_performance.invoke(
            {"sector_return": 12.5, "sp500_return": 10.0}
        )
        assert result == pytest.approx(2.5)


@pytest.mark.unit
class TestPeerDispersionTool:
    def test_basic(self):
        result = compute_peer_dispersion.invoke({"peer_returns": [10.0, 20.0, 30.0]})
        assert result is not None
        assert result == pytest.approx(8.165, rel=0.01)

    def test_too_few(self):
        result = compute_peer_dispersion.invoke({"peer_returns": [10.0, 20.0]})
        assert result is None
