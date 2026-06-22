"""
adx.py
------
Wilder ADX(14) on closed 4-hour candles.

Public API
----------
calc_adx(df_4h: pd.DataFrame) -> float
    Returns the most-recent ADX value (0–100).
    Matches MT4 / TradingView Wilder-smoothed ADX behaviour.

Requirements
------------
- df_4h must have a timezone-aware DatetimeIndex
- Columns required: 'high', 'low', 'close' (case-sensitive)
- Minimum 29 rows (2 × period + 1, where period = 14)
- Raises ValueError on any violated precondition
"""

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
_PERIOD = 14
_MIN_CANDLES = 2 * _PERIOD + 1  # 29


# ── helpers ──────────────────────────────────────────────────────────────────

def _wilder_rma(series: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's Running Moving Average (RMA / SMMA).

    Seed  : arithmetic mean of the first `period` values.
    Update: rma[i] = rma[i-1] * (period - 1) / period  +  series[i] / period

    Caller is responsible for passing a NaN-free slice starting at the
    first valid value.  Output length matches input length.
    Values before index (period - 1) are NaN (insufficient history).
    """
    n = len(series)
    out = np.full(n, np.nan)

    if n < period:
        return out

    # Seed: arithmetic mean of first `period` values
    seed = np.mean(series[:period])
    if np.isnan(seed):
        return out  # propagate failure cleanly

    out[period - 1] = seed
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = out[i - 1] * (1.0 - alpha) + series[i] * alpha

    return out


# ── public function ───────────────────────────────────────────────────────────

def calc_adx(df_4h: pd.DataFrame) -> float:
    """
    Calculate Wilder ADX(14) from closed 4-hour candles.

    Parameters
    ----------
    df_4h : pd.DataFrame
        Must have:
        - A timezone-aware DatetimeIndex (UTC or any fixed offset)
        - Columns 'high', 'low', 'close'
        - At least 29 rows (2 × 14 + 1)
        - No NaN values in required columns
        - Valid OHLC relationships: high >= low, high >= close, low <= close

    Returns
    -------
    float
        Most-recent ADX value in the range [0, 100].

    Raises
    ------
    ValueError
        - df_4h is not a DataFrame
        - Index is not a DatetimeIndex
        - Index is timezone-naive
        - Required columns are missing
        - Fewer than 29 rows
        - Any required column contains NaN
        - Any row violates OHLC sanity (high < low, high < close, low > close)
    """

    # ── structural validation ─────────────────────────────────────────────────

    if not isinstance(df_4h, pd.DataFrame):
        raise ValueError(
            f"df_4h must be a pandas DataFrame, got {type(df_4h).__name__}."
        )

    if not isinstance(df_4h.index, pd.DatetimeIndex):
        raise ValueError(
            "df_4h must have a DatetimeIndex. "
            f"Got {type(df_4h.index).__name__}."
        )

    if df_4h.index.tz is None:
        raise ValueError(
            "df_4h index must be timezone-aware (e.g. UTC). "
            "Localize with df.index = df.index.tz_localize('UTC') first."
        )

    required_cols = {"high", "low", "close"}
    missing = required_cols - set(df_4h.columns)
    if missing:
        raise ValueError(
            f"df_4h is missing required column(s): {sorted(missing)}. "
            f"Present columns: {list(df_4h.columns)}."
        )

    if len(df_4h) < _MIN_CANDLES:
        raise ValueError(
            f"calc_adx requires at least {_MIN_CANDLES} candles "
            f"(2 × period + 1 = 2 × {_PERIOD} + 1). "
            f"Got {len(df_4h)}."
        )

    # FIX 2a: sort by index immediately after structural validation,
    # before NaN checks and before any array extraction.
    df_4h = df_4h.sort_index()

    # ── NaN check ─────────────────────────────────────────────────────────────

    for col in required_cols:
        if df_4h[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains NaN values. "
                "Clean or forward-fill before calling calc_adx."
            )

    # FIX 2b: OHLC sanity check ───────────────────────────────────────────────
    # Every row must satisfy: high >= low, high >= close, low <= close.
    high_arr  = df_4h["high"].to_numpy(dtype=float)
    low_arr   = df_4h["low"].to_numpy(dtype=float)
    close_arr = df_4h["close"].to_numpy(dtype=float)

    bad_hl = np.where(high_arr < low_arr)[0]
    if len(bad_hl) > 0:
        i = bad_hl[0]
        raise ValueError(
            f"OHLC sanity violation at row index {i} "
            f"(timestamp={df_4h.index[i]}): "
            f"high ({high_arr[i]}) < low ({low_arr[i]})."
        )

    bad_hc = np.where(high_arr < close_arr)[0]
    if len(bad_hc) > 0:
        i = bad_hc[0]
        raise ValueError(
            f"OHLC sanity violation at row index {i} "
            f"(timestamp={df_4h.index[i]}): "
            f"high ({high_arr[i]}) < close ({close_arr[i]})."
        )

    bad_lc = np.where(low_arr > close_arr)[0]
    if len(bad_lc) > 0:
        i = bad_lc[0]
        raise ValueError(
            f"OHLC sanity violation at row index {i} "
            f"(timestamp={df_4h.index[i]}): "
            f"low ({low_arr[i]}) > close ({close_arr[i]})."
        )

    # ── extract arrays ────────────────────────────────────────────────────────
    # high_arr, low_arr, close_arr already extracted above; reuse them.
    high  = high_arr
    low   = low_arr
    close = close_arr
    n     = len(high)

    # ── step 1: True Range ────────────────────────────────────────────────────
    # TR[0] has no prior close → use high[0] - low[0]
    # TR[i] = max(high-low, |high - prev_close|, |low - prev_close|)

    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl  = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i]  - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    # ── step 2: Directional Movement ──────────────────────────────────────────
    # +DM[i]: upward move only when it exceeds the downward move
    # -DM[i]: downward move only when it exceeds the upward move
    # Strict greater-than (ties → both 0), matches MT4 behaviour.
    # Index 0: no prior bar → both 0.

    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        up   = high[i] - high[i - 1]
        down = low[i - 1] - low[i]

        if up > down and up > 0:
            plus_dm[i] = up

        if down > up and down > 0:
            minus_dm[i] = down

    # ── step 3: Wilder RMA of TR, +DM, -DM ───────────────────────────────────

    rma_tr       = _wilder_rma(tr,       _PERIOD)
    rma_plus_dm  = _wilder_rma(plus_dm,  _PERIOD)
    rma_minus_dm = _wilder_rma(minus_dm, _PERIOD)

    # ── step 4: DI lines ──────────────────────────────────────────────────────
    # +DI = 100 × RMA(+DM) / RMA(TR)
    # -DI = 100 × RMA(-DM) / RMA(TR)
    # Valid from index (period - 1) = 13 onward.

    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di  = np.where(rma_tr != 0, 100.0 * rma_plus_dm  / rma_tr, 0.0)
        minus_di = np.where(rma_tr != 0, 100.0 * rma_minus_dm / rma_tr, 0.0)

    # ── step 5: DX ────────────────────────────────────────────────────────────
    # DX = 100 × |+DI - -DI| / (+DI + -DI)

    di_sum  = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)

    with np.errstate(invalid="ignore", divide="ignore"):
        dx_full = np.where(di_sum != 0, 100.0 * di_diff / di_sum, 0.0)

    # ── step 6: ADX = Wilder RMA of DX ───────────────────────────────────────
    # Slice dx to only the valid region [period-1 :] before seeding RMA,
    # so the seed is computed from real DX values only.
    #
    # First valid DX index : period - 1 = 13
    # First valid ADX index: 2 * period - 2 = 26
    # With _MIN_CANDLES = 29 the last index is 28, so index 26 always exists.

    dx_start = _PERIOD - 1                 # index 13
    dx_valid = dx_full[dx_start:]          # NaN-free slice

    adx_valid = _wilder_rma(dx_valid, _PERIOD)

    # Re-embed into full-length array
    adx_full = np.full(n, np.nan)
    adx_full[dx_start:] = adx_valid

    last_adx = adx_full[-1]

    if np.isnan(last_adx):
        raise ValueError(
            "ADX calculation produced NaN for the last candle despite "
            "passing the minimum-candle check. Please file a bug report."
        )

    return float(last_adx)