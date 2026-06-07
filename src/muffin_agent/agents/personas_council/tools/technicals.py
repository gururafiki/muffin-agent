"""Technical-analysis indicators and ensemble signals.

Pure-pandas/numpy implementations of the 5-strategy technical ensemble
ported from ai-hedge-fund's ``src/agents/technicals.py``.  Each strategy
returns a deterministic ``StrategyResult`` (3-tier signal + confidence +
metrics dict); the :func:`combine_technical_signals` helper aggregates
them into one weighted directional view.

These are **not** ``@tool``-decorated — they're called inline by the
``technical_analysis_node`` (Phase 3.1) and by any persona that wants
technical evidence (Druckenmiller for momentum, Taleb implicitly for
volatility regime).  The technical_analysis_node maps the 3-tier internal
vocabulary to the 5-tier ``InvestmentSignal`` when emitting an
:class:`muffin_agent.agents.personas_council.schemas.AnalystSignal`.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd

TacticalSignal = Literal["bullish", "bearish", "neutral"]
"""3-tier internal signal vocabulary used by individual technical strategies.
The technical_analysis_node maps this to the 5-tier ``InvestmentSignal``
when emitting an ``AnalystSignal``."""


class StrategyResult(TypedDict):
    """Output of one technical strategy."""

    signal: TacticalSignal
    confidence: float  # 0.0–1.0
    metrics: dict[str, float | None]


# ── Atomic indicators ─────────────────────────────────────────────────────────


def _ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average over ``close`` with the given ``span``."""
    return close.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (period default 14).

    Implementation matches ai-hedge-fund's: separate gains/losses, simple
    rolling mean (not Wilder's EMA smoothing — kept for parity).
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Upper / lower Bollinger Bands at ±``num_std`` over a rolling ``window``."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return sma + std * num_std, sma - std * num_std


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index over OHLC data.

    Args:
        df: Must contain ``high``, ``low``, ``close`` columns.
        period: Smoothing period (default 14).

    Returns:
        pd.Series with the ADX values (0–100 scale).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0))
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    )
    plus_dm.index = df.index
    minus_dm.index = df.index
    tr_ema = tr.ewm(span=period).mean()
    plus_di = 100 * plus_dm.ewm(span=period).mean() / tr_ema
    minus_di = 100 * minus_dm.ewm(span=period).mean() / tr_ema
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range over OHLC data; rolling-mean smoothing (period default 14)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _hurst_exponent(prices: pd.Series, max_lag: int = 20) -> float:
    """Hurst exponent via R/S log-log regression.

    H < 0.5 → mean reverting.  H = 0.5 → random walk.  H > 0.5 → trending.
    Returns 0.5 (random walk) if regression is degenerate.
    """
    arr = np.asarray(prices.dropna(), dtype=float)
    if len(arr) < max_lag + 1:
        return 0.5
    lags = range(2, max_lag)
    tau = [max(1e-8, math.sqrt(float(np.std(arr[lag:] - arr[:-lag])))) for lag in lags]
    try:
        reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(reg[0])
    except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
        return 0.5


def _safe_float(x: Any) -> float | None:
    """Coerce *x* to float; return None on NaN / inf / non-numeric."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _df_from_bars(bars: list[dict[str, Any]]) -> pd.DataFrame:
    """Build an OHLCV DataFrame from a list of bar dicts.

    Accepts dicts with keys ``open``, ``high``, ``low``, ``close``,
    ``volume``, plus an optional ``date``.  Sorts by date when present.
    Used by callers that have a ``PersonaDataBundle.prices_1y`` list.
    """
    df = pd.DataFrame(bars)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Strategy 1: trend following ───────────────────────────────────────────────


def compute_trend_signal(bars: list[dict[str, Any]]) -> StrategyResult:
    """Multi-timeframe trend signal with ADX-weighted confidence.

    Bullish when EMA(8) > EMA(21) AND EMA(21) > EMA(55).  Bearish when both
    are negated.  Else neutral.  Confidence = ADX / 100 (so a weak trend
    on aligned EMAs still produces a weak signal).
    """
    if len(bars) < 60:
        return _neutral_with_message("Insufficient price history for trend")
    df = _df_from_bars(bars)
    close = df["close"]
    ema_8 = _ema(close, 8)
    ema_21 = _ema(close, 21)
    ema_55 = _ema(close, 55)
    adx_series = _adx(df, 14)

    short_up = bool(ema_8.iloc[-1] > ema_21.iloc[-1])
    medium_up = bool(ema_21.iloc[-1] > ema_55.iloc[-1])
    trend_strength = (
        float(adx_series.iloc[-1]) / 100.0
        if not math.isnan(adx_series.iloc[-1])
        else 0.5
    )

    if short_up and medium_up:
        signal: TacticalSignal = "bullish"
        confidence = trend_strength
    elif not short_up and not medium_up:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": max(0.0, min(1.0, confidence)),
        "metrics": {
            "ema_8": _safe_float(ema_8.iloc[-1]),
            "ema_21": _safe_float(ema_21.iloc[-1]),
            "ema_55": _safe_float(ema_55.iloc[-1]),
            "adx": _safe_float(adx_series.iloc[-1]),
            "trend_strength": _safe_float(trend_strength),
        },
    }


# ── Strategy 2: mean reversion ────────────────────────────────────────────────


def compute_mean_reversion_signal(bars: list[dict[str, Any]]) -> StrategyResult:
    """Mean-reversion signal via 50-day z-score + Bollinger position + RSI.

    Bullish when ``z_score < -2`` AND price near lower BB (``price_vs_bb < 0.2``);
    bearish on the symmetric upper-band condition; else neutral.
    Confidence = ``min(|z_score| / 4, 1.0)``.
    """
    if len(bars) < 50:
        return _neutral_with_message("Insufficient price history for mean reversion")
    df = _df_from_bars(bars)
    close = df["close"]
    ma_50 = close.rolling(50).mean()
    std_50 = close.rolling(50).std()
    z = (close - ma_50) / std_50
    bb_upper, bb_lower = _bollinger_bands(close, 20)
    rsi_14 = _rsi(close, 14)
    rsi_28 = _rsi(close, 28)

    last_close = float(close.iloc[-1])
    last_z = float(z.iloc[-1])
    upper = float(bb_upper.iloc[-1])
    lower = float(bb_lower.iloc[-1])
    if upper == lower:
        price_vs_bb = 0.5
    else:
        price_vs_bb = (last_close - lower) / (upper - lower)

    if last_z < -2 and price_vs_bb < 0.2:
        signal: TacticalSignal = "bullish"
        confidence = min(abs(last_z) / 4, 1.0)
    elif last_z > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(last_z) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": _safe_float(last_z),
            "price_vs_bb": _safe_float(price_vs_bb),
            "rsi_14": _safe_float(rsi_14.iloc[-1]),
            "rsi_28": _safe_float(rsi_28.iloc[-1]),
        },
    }


# ── Strategy 3: momentum ──────────────────────────────────────────────────────


def compute_momentum_signal(bars: list[dict[str, Any]]) -> StrategyResult:
    """Multi-timeframe momentum signal with volume confirmation.

    Momentum score = 0.4·MOM_1m + 0.3·MOM_3m + 0.3·MOM_6m (sum-of-returns
    over 21 / 63 / 126 sessions).  Volume confirmation requires the
    current volume to exceed its 21-session SMA.

    Bullish when score > 5% AND volume confirms.  Bearish when < -5% AND
    volume confirms.  Else neutral.  Confidence = ``min(5·|score|, 1.0)``.
    """
    if len(bars) < 126:
        return _neutral_with_message("Insufficient price history for momentum")
    df = _df_from_bars(bars)
    close = df["close"]
    volume = df.get("volume")
    returns = close.pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()
    score = 0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m

    if volume is not None and len(volume.dropna()) >= 21:
        volume_ma = volume.rolling(21).mean()
        vol_ratio = float(volume.iloc[-1]) / float(volume_ma.iloc[-1])
        volume_confirmed = vol_ratio > 1.0
    else:
        vol_ratio = 1.0
        volume_confirmed = True  # no volume data — give momentum the benefit

    last_score = float(score.iloc[-1])
    if last_score > 0.05 and volume_confirmed:
        signal: TacticalSignal = "bullish"
        confidence = min(abs(last_score) * 5, 1.0)
    elif last_score < -0.05 and volume_confirmed:
        signal = "bearish"
        confidence = min(abs(last_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": _safe_float(mom_1m.iloc[-1]),
            "momentum_3m": _safe_float(mom_3m.iloc[-1]),
            "momentum_6m": _safe_float(mom_6m.iloc[-1]),
            "volume_ratio": _safe_float(vol_ratio),
        },
    }


# ── Strategy 4: volatility regime ─────────────────────────────────────────────


def compute_volatility_regime_signal(bars: list[dict[str, Any]]) -> StrategyResult:
    """Volatility regime signal.

    Annualised 21-day vol vs its 63-day MA → ``vol_regime``.  Bullish
    when ``vol_regime < 0.8`` AND ``vol_z < -1`` (compressed vol, expansion
    likely).  Bearish on the symmetric high-vol condition.

    Confidence = ``min(|vol_z| / 3, 1.0)``.
    """
    if len(bars) < 84:  # need 21 + 63 for the rolling-of-rolling
        return _neutral_with_message("Insufficient price history for vol regime")
    df = _df_from_bars(bars)
    returns = df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma
    vol_z = (hist_vol - vol_ma) / hist_vol.rolling(63).std()
    atr = _atr(df, 14)
    atr_ratio = atr / df["close"]

    current_regime = float(vol_regime.iloc[-1])
    current_z = float(vol_z.iloc[-1])

    if current_regime < 0.8 and current_z < -1:
        signal: TacticalSignal = "bullish"
        confidence = min(abs(current_z) / 3, 1.0)
    elif current_regime > 1.2 and current_z > 1:
        signal = "bearish"
        confidence = min(abs(current_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": _safe_float(hist_vol.iloc[-1]),
            "volatility_regime": _safe_float(current_regime),
            "volatility_z_score": _safe_float(current_z),
            "atr_ratio": _safe_float(atr_ratio.iloc[-1]),
        },
    }


# ── Strategy 5: statistical arbitrage ─────────────────────────────────────────


def compute_stat_arb_signal(bars: list[dict[str, Any]]) -> StrategyResult:
    """Statistical-arbitrage signal via Hurst exponent + return distribution.

    Bullish when ``hurst < 0.4`` (mean-reverting) AND 63-day skew > 1
    (recent positive tail).  Bearish when ``hurst < 0.4`` AND skew < -1.
    Else neutral.  Confidence = ``(0.5 - hurst) × 2``.
    """
    if len(bars) < 70:
        return _neutral_with_message("Insufficient price history for stat arb")
    df = _df_from_bars(bars)
    returns = df["close"].pct_change()
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()
    hurst = _hurst_exponent(df["close"])
    last_skew = float(skew.iloc[-1])

    if hurst < 0.4 and last_skew > 1:
        signal: TacticalSignal = "bullish"
        confidence = max(0.0, min((0.5 - hurst) * 2, 1.0))
    elif hurst < 0.4 and last_skew < -1:
        signal = "bearish"
        confidence = max(0.0, min((0.5 - hurst) * 2, 1.0))
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": _safe_float(hurst),
            "skewness": _safe_float(last_skew),
            "kurtosis": _safe_float(kurt.iloc[-1]),
        },
    }


# ── Combination ───────────────────────────────────────────────────────────────


DEFAULT_STRATEGY_WEIGHTS: dict[str, float] = {
    "trend": 0.25,
    "mean_reversion": 0.20,
    "momentum": 0.25,
    "volatility": 0.15,
    "stat_arb": 0.15,
}
"""Strategy weights matching ai-hedge-fund's upstream `combine_technical_signals`
defaults.  Sum to 1.0; callers may pass overrides."""


def combine_technical_signals(
    results: dict[str, StrategyResult],
    weights: dict[str, float] | None = None,
) -> StrategyResult:
    """Weighted-confidence ensemble across strategy results.

    Numericalises each result's signal (bullish=+1 / neutral=0 / bearish=-1),
    weights it by ``weight × confidence``, sums, and normalises.  Final
    threshold: ``+0.2`` → bullish, ``-0.2`` → bearish, else neutral.
    Returned confidence is ``|final_score|``.

    Args:
        results: Mapping from strategy name to its ``StrategyResult``.
            Strategy names should match keys in *weights*.
        weights: Optional override; defaults to :data:`DEFAULT_STRATEGY_WEIGHTS`.
            Strategies present in *results* but absent from *weights* are
            skipped (no contribution).

    Returns:
        ``StrategyResult`` whose ``metrics`` exposes per-strategy
        ``contribution_<name>`` values for debugging.
    """
    weights = weights or DEFAULT_STRATEGY_WEIGHTS
    signal_to_num: dict[TacticalSignal, int] = {
        "bullish": 1,
        "neutral": 0,
        "bearish": -1,
    }
    weighted_sum = 0.0
    total_weight = 0.0
    contributions: dict[str, float | None] = {}
    for name, r in results.items():
        if name not in weights:
            continue
        w = weights[name]
        conf = r["confidence"]
        num = signal_to_num[r["signal"]]
        contribution = num * w * conf
        weighted_sum += contribution
        total_weight += w * conf
        contributions[f"contribution_{name}"] = contribution

    final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    if final_score > 0.2:
        signal: TacticalSignal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "confidence": min(abs(final_score), 1.0),
        "metrics": {
            "final_score": _safe_float(final_score),
            **contributions,
        },
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _neutral_with_message(message: str) -> StrategyResult:
    """Return a neutral ``StrategyResult`` whose ``metrics["note"]`` is *message*.

    Used when a strategy cannot run (insufficient data).  Confidence is
    set to 0.0 — explicitly distinct from a genuinely-mixed neutral
    (confidence 0.5) so the ensemble doesn't get pulled by missing data.
    """
    return {
        "signal": "neutral",
        "confidence": 0.0,
        "metrics": {"note": message},  # type: ignore[dict-item]
    }
