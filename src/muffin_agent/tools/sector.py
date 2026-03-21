"""Sector and peer comparison tools — relative performance, dispersion."""

from __future__ import annotations

from langchain_core.tools import tool


@tool(parse_docstring=True)
def compute_sector_relative_performance(
    sector_return: float,
    sp500_return: float,
) -> float:
    """Compute sector performance relative to S&P 500.

    Args:
        sector_return: Sector total return in percentage points.
        sp500_return: S&P 500 total return in percentage points.

    Returns:
        Difference in percentage points (positive = outperformance).
    """
    return sector_return - sp500_return


@tool(parse_docstring=True)
def compute_peer_dispersion(
    peer_returns: list[float],
) -> float | None:
    """Compute standard deviation of peer stock returns.

    Measure alpha opportunity via peer return dispersion.
    Requires at least 3 peers.

    Args:
        peer_returns: List of peer company returns in percentage points.
            Minimum 3 values required.

    Returns:
        Standard deviation of returns in percentage points, or None if
        fewer than 3 peers provided.
    """
    if len(peer_returns) < 3:
        return None
    mean = sum(peer_returns) / len(peer_returns)
    variance = sum((r - mean) ** 2 for r in peer_returns) / len(peer_returns)
    return variance**0.5
