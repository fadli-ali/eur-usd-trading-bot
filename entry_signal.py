# entry_signal.py
"""
entry_signal.py
---------------
Signal evaluation for the EUR/USD mean-reversion bot.

Public API
----------
evaluate_signal(df_1h, df_4h, current_time, events, daily_open) -> dict | None

Returns a signal dict when all entry conditions are met, or None otherwise.
All candle data is filtered to current_time before use to prevent look-ahead bias.

Signal dict keys
----------------
signal        : 'long' or 'short'
entry_price   : float  - last closed 1h candle close price
stop_price    : float  - stop level (band edge + 0.2 x ATR beyond entry)
target_price  : float  - daily open (mean-reversion target)
atr           : float  - 10-day ATR used for band calculation
band_upper    : float  - daily_open + 0.4 x ATR
band_lower    : float  - daily_open - 0.4 x ATR
adx           : float  - most-recent 4h ADX(14)
rsi           : float  - most-recent 1h RSI(14)
"""

from __future__ import annotations

import math
import numbers
from datetime import datetime
from typing import Optional

import pandas as pd

from atr_rsi import calc_atr, calc_rsi
from adx import calc_adx
from news_filter import is_news_blackout

_ATR_BAND_MULT = 0.4
_ATR_STOP_MULT = 0.2
_ADX_THRESHOLD = 25.0
_RSI_LONG_MAX  = 35.0
_RSI_SHORT_MIN = 65.0


def _validate_daily_open(daily_open: object) -> float:
    if not isinstance(daily_open, numbers.Real):
        raise ValueError(f"daily_open must be a real number, got {type(daily_open).__name__!r}.")
    if not math.isfinite(float(daily_open)):
        raise ValueError(f"daily_open must be finite, got {daily_open!r}.")
    if daily_open <= 0:
        raise ValueError(f"daily_open must be > 0, got {daily_open!r}.")
    return float(daily_open)


def _validate_current_time(current_time: object) -> datetime:
    if not isinstance(current_time, datetime):
        raise ValueError(f"current_time must be a datetime, got {type(current_time).__name__!r}.")
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("current_time must be timezone-aware.")
    return current_time


def _validate_df(df: object, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(df).__name__!r}.")
    if df.empty:
        raise ValueError(f"{name} must not be empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError(f"{name} index must be timezone-aware.")
    return df.sort_index()


def evaluate_signal(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    current_time: datetime,
    events: list,
    daily_open: float,
) -> Optional[dict]:
    """
    Evaluate entry conditions and return a signal dict or None.

    All candle data is filtered to candles at or before current_time before
    any indicator is computed, preventing look-ahead bias in backtesting.

    Parameters
    ----------
    df_1h : pd.DataFrame
        Closed 1h OHLC candles. Timezone-aware DatetimeIndex.
        Columns: open, high, low, close.
    df_4h : pd.DataFrame
        Closed 4h OHLC candles. Timezone-aware DatetimeIndex.
        Columns: high, low, close.
    current_time : datetime
        Current wall-clock time. Must be tz-aware.
    events : list[dict]
        Economic calendar events for is_news_blackout.
    daily_open : float
        Today's FX day open price. Must be finite and > 0.

    Returns
    -------
    dict or None
    """
    # -- validate inputs --
    current_time = _validate_current_time(current_time)
    daily_open   = _validate_daily_open(daily_open)
    if not isinstance(events, list):
        raise ValueError(f"events must be a list, got {type(events).__name__!r}.")
    df_1h = _validate_df(df_1h, "df_1h")
    df_4h = _validate_df(df_4h, "df_4h")

    # -- filter to closed candles only (prevents look-ahead bias) --
    ts = pd.Timestamp(current_time)
    df_1h_closed = df_1h[df_1h.index <= ts]
    df_4h_closed = df_4h[df_4h.index <= ts]

    if df_1h_closed.empty:
        raise ValueError("df_1h has no closed candles at or before current_time.")
    if df_4h_closed.empty:
        raise ValueError("df_4h has no closed candles at or before current_time.")

    # -- (1) news blackout filter --
    if is_news_blackout(current_time, events):
        return None

    # -- (2) ADX regime filter --
    adx = calc_adx(df_4h_closed)
    if adx >= _ADX_THRESHOLD:
        return None

    # -- (3) ATR and RSI --
    atr = calc_atr(df_1h_closed, current_time)
    rsi = calc_rsi(df_1h_closed)

    # -- (4) bands --
    band_upper = daily_open + _ATR_BAND_MULT * atr
    band_lower = daily_open - _ATR_BAND_MULT * atr

    # -- (5) last closed 1h candle price --
    entry_price = float(df_1h_closed["close"].iloc[-1])

    # -- (6) long signal --
    if entry_price <= band_lower and rsi < _RSI_LONG_MAX:
        stop_price = band_lower - _ATR_STOP_MULT * atr
        return {
            "signal":       "long",
            "entry_price":  entry_price,
            "stop_price":   round(stop_price, 5),
            "target_price": daily_open,
            "atr":          atr,
            "band_upper":   round(band_upper, 5),
            "band_lower":   round(band_lower, 5),
            "adx":          adx,
            "rsi":          rsi,
        }

    # -- (7) short signal --
    if entry_price >= band_upper and rsi > _RSI_SHORT_MIN:
        stop_price = band_upper + _ATR_STOP_MULT * atr
        return {
            "signal":       "short",
            "entry_price":  entry_price,
            "stop_price":   round(stop_price, 5),
            "target_price": daily_open,
            "atr":          atr,
            "band_upper":   round(band_upper, 5),
            "band_lower":   round(band_lower, 5),
            "adx":          adx,
            "rsi":          rsi,
        }

    # -- (8) no signal --
    return None
