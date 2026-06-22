# atr_rsi.py
"""
atr_rsi.py - ATR and RSI calculations for the EUR/USD mean-reversion bot.

Function 1: calc_atr(df_1h, current_time)
    Wilder ATR over 10 completed FX trading days (5pm NY to 5pm NY).

    Method (matches strategy spec "0.4 x 10-day ATR"):
      1. Aggregate 1h candles into FX-day daily bars.
      2. Drop the in-progress FX day.
      3. Take the 10 most-recent completed FX days.
      4. Compute daily True Range for each of those 10 days using
         prev_daily_close from the full completed daily series (not
         just the selected window), so the oldest selected day gets
         a proper TR if a prior day exists.
         Day with no prior daily close: TR = daily_high - daily_low.
      5. ATR = mean of those 10 daily TR values (Wilder seed, period=10).
         Exactly 10 values selected so seed IS the result.
    Returns float (ATR in price units, e.g. 0.00850 = 85 pips).

Function 2: calc_rsi(df_1h)
    RSI(14) using Wilder smoothing on closed 1h candles.
    Returns float in [0, 100].
    Minimum 42 candles required (3 x period) for stable reading.

Both functions:
    - Require a tz-aware DatetimeIndex; raise ValueError on naive datetimes.
    - Raise ValueError on insufficient history or malformed input.
    - Do not mutate the input DataFrame.
    - Expect df_1h columns: open, high, low, close.
"""

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

ATR_FX_DAYS     = 10
RSI_PERIOD      = 14
RSI_MIN_CANDLES = RSI_PERIOD * 3   # 42


def _validate_df(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"DataFrame is missing required column: '{col}'. "
                f"Required: {required_cols}."
            )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError(
            "DataFrame index must be timezone-aware. "
            "Convert with: df.index = df.index.tz_localize('UTC')"
        )
    if df.empty:
        raise ValueError("DataFrame is empty.")
    return df.sort_index()


def _to_aware_timestamp(dt, name: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(dt)
    except Exception as exc:
        raise ValueError(f"'{name}' could not be converted to a Timestamp: {exc}") from exc
    if ts.tzinfo is None:
        raise ValueError(
            f"'{name}' must be timezone-aware. "
            f"Example: pd.Timestamp('2024-01-15 10:00', tz='UTC'). Got: {dt!r}"
        )
    return ts


def _fx_day_start(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Return the start of the FX day containing ts.
    FX day: 17:00 NY to 17:00 NY next calendar date.
    """
    ts_ny  = ts.astimezone(NY_TZ)
    cutoff = ts_ny.replace(hour=17, minute=0, second=0, microsecond=0)
    if ts_ny >= cutoff:
        return cutoff
    return cutoff - pd.Timedelta(days=1)


def _build_fx_daily_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sorted 1h OHLC candles into FX-day daily bars.
    Returns one row per FX day indexed by FX day start (17:00 NY).
    """
    index_ny  = df.index.tz_convert(NY_TZ)
    fx_labels = pd.DatetimeIndex(
        [_fx_day_start(ts) for ts in index_ny],
        dtype="datetime64[ns, America/New_York]",
    )
    df2          = df.copy()
    df2["_fxday"] = fx_labels
    g            = df2.groupby("_fxday", sort=True)
    daily        = pd.DataFrame({
        "daily_open":  g["open"].first(),
        "daily_high":  g["high"].max(),
        "daily_low":   g["low"].min(),
        "daily_close": g["close"].last(),
    })
    daily.index.name = "fx_day_start"
    return daily


def calc_atr(df_1h: pd.DataFrame, current_time) -> float:
    """
    Wilder ATR(10) computed from 10 completed FX-day daily bars.

    Parameters
    ----------
    df_1h : pd.DataFrame
        Closed 1h OHLC candles. Timezone-aware DatetimeIndex.
        Columns: open, high, low, close.
    current_time : datetime-like (tz-aware)
        Used to determine which FX day is in-progress (excluded).

    Returns
    -------
    float  ATR in price units (e.g. 0.0085 = 85 pips).

    Raises
    ------
    ValueError on bad input or fewer than 10 completed FX days.
    """
    current_time = _to_aware_timestamp(current_time, "current_time")
    df           = _validate_df(df_1h, ["open", "high", "low", "close"])

    current_fx_day = _fx_day_start(current_time)

    # Build ALL completed daily bars (exclude in-progress day)
    all_daily = _build_fx_daily_bars(df)
    completed  = all_daily[all_daily.index < current_fx_day]

    if len(completed) < ATR_FX_DAYS:
        raise ValueError(
            f"calc_atr requires {ATR_FX_DAYS} completed FX days; "
            f"only {len(completed)} found before {current_time}."
        )

    # Take the last 10 completed days
    window = completed.iloc[-ATR_FX_DAYS:]

    # Compute daily TR using prev_daily_close from the FULL completed series
    # so the oldest selected day gets a real prior close if one exists.
    prev_close_series = completed["daily_close"].shift(1)

    dh = window["daily_high"].values.astype(float)
    dl = window["daily_low"].values.astype(float)
    dc = window["daily_close"].values.astype(float)
    pc = prev_close_series.reindex(window.index).values.astype(float)

    tr = np.empty(ATR_FX_DAYS)
    for i in range(ATR_FX_DAYS):
        if np.isnan(pc[i]):
            # No prior daily close available: use high-low only
            tr[i] = dh[i] - dl[i]
        else:
            tr[i] = max(
                dh[i] - dl[i],
                abs(dh[i] - pc[i]),
                abs(dl[i] - pc[i]),
            )

    # Wilder ATR(10) seed = mean of exactly 10 daily TRs
    atr = float(np.mean(tr))
    return atr


def calc_rsi(df_1h: pd.DataFrame) -> float:
    """
    RSI(14) using Wilder smoothing on closed 1-hour candles.

    Wilder smoothing:
      seed    = SMA of first 14 up-moves / down-moves
      update  = avg = avg * (13/14) + value * (1/14)
      minimum = 42 candles (3 x period)

    Parameters
    ----------
    df_1h : pd.DataFrame
        Closed 1h OHLC candles. Timezone-aware DatetimeIndex.
        Columns: open, high, low, close.

    Returns
    -------
    float  RSI in [0.0, 100.0].

    Raises
    ------
    ValueError on bad input or fewer than 42 candles.
    """
    df = _validate_df(df_1h, ["open", "high", "low", "close"])

    if len(df) < RSI_MIN_CANDLES:
        raise ValueError(
            f"calc_rsi requires at least {RSI_MIN_CANDLES} closed candles "
            f"(3 x RSI period {RSI_PERIOD}); only {len(df)} provided."
        )

    close  = df["close"].to_numpy(dtype=float)
    delta  = np.diff(close)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = float(np.mean(gains[:RSI_PERIOD]))
    avg_loss = float(np.mean(losses[:RSI_PERIOD]))

    for i in range(RSI_PERIOD, len(delta)):
        avg_gain = (avg_gain * (RSI_PERIOD - 1) + gains[i])  / RSI_PERIOD
        avg_loss = (avg_loss * (RSI_PERIOD - 1) + losses[i]) / RSI_PERIOD

    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0

    rs  = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N   = 15 * 24
    idx = pd.date_range("2024-01-02 17:00", periods=N, freq="1h", tz="America/New_York")
    p   = 1.08
    closes = []
    for _ in range(N):
        p += rng.normal(0, 0.0005)
        closes.append(p)
    closes = np.array(closes)
    df = pd.DataFrame({
        "open":  closes - 0.0001,
        "high":  closes + 0.0003,
        "low":   closes - 0.0003,
        "close": closes,
    }, index=idx)

    current_time = idx[-1] + pd.Timedelta(minutes=30)
    atr = calc_atr(df, current_time)
    print(f"ATR (10 FX days, daily bars): {atr:.6f}  ({atr*10000:.1f} pips)")
    assert isinstance(atr, float) and atr > 0

    rsi = calc_rsi(df)
    print(f"RSI(14): {rsi:.4f}")
    assert isinstance(rsi, float) and 0 <= rsi <= 100

    print("All smoke tests passed.")
