# trade_logger.py
"""
Trade logger for the EUR/USD mean-reversion bot.
Appends one row per closed trade to trades.csv.

Platform: Linux / macOS only (uses fcntl). Not compatible with Windows.

P&L Convention
--------------
OANDA fill prices embed the spread: longs fill at ask, shorts fill at bid.
Fill-to-fill P&L therefore already has spread drag deducted.

    gross_pnl   = (exit_fill - entry_fill) x side_sign x units  [USD]
    spread_cost = spread_pips x 0.0001 x units                  [USD, audit only]
    net_pnl     = gross_pnl

spread_cost is an audit column only — do not subtract it again.
If explicit commissions are added later, subtract them from net_pnl here.
Callers MUST pass actual OANDA fill prices, not mid prices.

Timestamp convention
--------------------
All timestamps stored as UTC, ISO-8601: YYYY-MM-DDTHH:MM:SSZ.
Naive datetimes are rejected with ValueError.
Aware non-UTC datetimes are converted to UTC before storage.
"""

import csv
import fcntl
import logging
import math
import numbers
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH: Path = Path(os.environ.get("TRADE_LOG_PATH", "trades.csv"))

FIELDNAMES: list[str] = [
    "timestamp",
    "pair",
    "side",
    "entry_price",
    "exit_price",
    "pips",
    "gross_pnl",
    "spread_cost",
    "net_pnl",
    "exit_reason",
]

SUPPORTED_PAIRS: set[str] = {"EUR_USD"}
PIP_FACTOR   = 10_000
PIP_DECIMALS = 1
PNL_DECIMALS = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trade_logger")


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------

def _validate_timestamp(ts: datetime) -> datetime:
    """
    Reject non-datetime objects and naive datetimes.
    Convert any tz-aware datetime to UTC.
    Returns the UTC datetime ready for formatting.
    """
    # FIX 1a: isinstance guard — touch .tzinfo only after confirming type
    if not isinstance(ts, datetime):
        raise ValueError(
            f"timestamp must be a datetime object, got {type(ts).__name__!r}."
        )
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(
            "timestamp must be timezone-aware. "
            "Use datetime.now(timezone.utc) or attach a tzinfo."
        )
    return ts.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _check_finite_real(value: object, name: str) -> None:
    """
    FIX 1b: Raise ValueError if value is not a real number or is NaN/inf.
    Accepts any numbers.Real subclass (int, float, numpy scalar, Decimal, …).
    """
    if not isinstance(value, numbers.Real):
        raise ValueError(
            f"{name} must be a real number, got {type(value).__name__!r}."
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"{name} must be a finite number (not NaN or inf), got {value!r}."
        )


def _validate_inputs(
    pair: str,
    side: str,
    entry_price: float,
    exit_price: float,
    units: float,
    spread_pips: float,
    exit_reason: str,
) -> None:
    """Raise ValueError with a descriptive message on any invalid input."""
    if pair not in SUPPORTED_PAIRS:
        raise ValueError(
            f"pair={pair!r} not supported. Supported: {SUPPORTED_PAIRS}"
        )
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    # Numeric guards: type + finiteness first, then domain checks
    _check_finite_real(entry_price,  "entry_price")
    _check_finite_real(exit_price,   "exit_price")
    _check_finite_real(units,        "units")
    _check_finite_real(spread_pips,  "spread_pips")

    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    if exit_price <= 0:
        raise ValueError(f"exit_price must be > 0, got {exit_price}")
    if units <= 0:
        raise ValueError(f"units must be > 0, got {units}")
    if spread_pips < 0:
        raise ValueError(f"spread_pips must be >= 0, got {spread_pips}")

    if not isinstance(exit_reason, str) or not exit_reason.strip():
        raise ValueError("exit_reason must be a non-empty string")


# ---------------------------------------------------------------------------
# P&L computation
# ---------------------------------------------------------------------------

def _compute_fields(
    side: str,
    entry_price: float,
    exit_price: float,
    units: float,
    spread_pips: float,
) -> dict:
    """
    Compute pips, gross_pnl, spread_cost, net_pnl for a EUR/USD trade.
    EUR/USD quote currency is USD so P&L is naturally in USD.
    """
    side_sign  = 1 if side == "long" else -1
    price_diff = exit_price - entry_price

    pips        = round(price_diff * PIP_FACTOR * side_sign, PIP_DECIMALS)
    gross_pnl   = round(price_diff * side_sign * units,       PNL_DECIMALS)
    spread_cost = round((spread_pips / PIP_FACTOR) * units,   PNL_DECIMALS)
    net_pnl     = gross_pnl

    return {
        "pips":        pips,
        "gross_pnl":   gross_pnl,
        "spread_cost": spread_cost,
        "net_pnl":     net_pnl,
    }


# ---------------------------------------------------------------------------
# Race-safe CSV write
# ---------------------------------------------------------------------------

def _write_row_locked(row: dict, csv_path: Path) -> None:
    """
    Append one CSV row under an exclusive fcntl lock.
    Lock file is <csv_path>.lock, co-located with the CSV.
    Header check and append happen inside the same lock acquisition.
    """
    lock_path = Path(str(csv_path) + ".lock")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            needs_header = (
                not csv_path.exists() or csv_path.stat().st_size == 0
            )
            with open(csv_path, "a", newline="") as csv_fh:
                writer = csv.DictWriter(
                    csv_fh,
                    fieldnames=FIELDNAMES,
                    extrasaction="raise",
                )
                if needs_header:
                    writer.writeheader()
                    logger.info("Initialised trade log: %s", csv_path.resolve())
                writer.writerow(row)
                csv_fh.flush()
                os.fsync(csv_fh.fileno())
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_trade(
    pair: str,
    side: Literal["long", "short"],
    entry_price: float,
    exit_price: float,
    units: float,
    spread_pips: float,
    exit_reason: str,
    timestamp: datetime | None = None,
    csv_path: Path = CSV_PATH,
) -> dict:
    """
    Validate inputs, compute metrics, append one row to trades.csv.

    Args:
        pair:         Instrument, e.g. 'EUR_USD'.
        side:         'long' or 'short'.
        entry_price:  OANDA ask fill at entry (long) or bid fill (short).
                      Must be a finite positive real number.
        exit_price:   OANDA bid fill at exit (long) or ask fill (short).
                      Must be a finite positive real number.
        units:        Units traded (positive finite real number).
        spread_pips:  Ask-bid spread at entry in pips (audit column only).
                      Must be a finite non-negative real number.
        exit_reason:  Non-empty label: 'target_hit', 'stop_hit',
                      'eod_close', 'regime_filter', etc.
        timestamp:    Timezone-aware close time; defaults to now(UTC).
                      Must be a datetime instance.
                      Naive datetimes raise ValueError.
                      Non-UTC aware datetimes are converted to UTC.
        csv_path:     Override default log path (useful in tests).

    Returns:
        The complete row dict written to CSV.

    Raises:
        ValueError: on invalid arguments, wrong types, NaN/inf numerics,
                    or naive timestamp.
        OSError:    on filesystem errors.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    timestamp_utc = _validate_timestamp(timestamp)

    _validate_inputs(
        pair, side, entry_price, exit_price, units, spread_pips, exit_reason
    )

    computed = _compute_fields(side, entry_price, exit_price, units, spread_pips)

    row = {
        "timestamp":   timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pair":        pair,
        "side":        side,
        "entry_price": round(entry_price, 5),
        "exit_price":  round(exit_price, 5),
        "pips":        computed["pips"],
        "gross_pnl":   computed["gross_pnl"],
        "spread_cost": computed["spread_cost"],
        "net_pnl":     computed["net_pnl"],
        "exit_reason": exit_reason.strip(),
    }

    _write_row_locked(row, csv_path)

    logger.info(
        "Trade logged | %s %s %s | entry=%.5f exit=%.5f | "
        "pips=%.1f gross=%.4f spread_cost=%.4f net=%.4f | reason=%s",
        row["timestamp"], pair, side,
        entry_price, exit_price,
        computed["pips"], computed["gross_pnl"],
        computed["spread_cost"], computed["net_pnl"],
        row["exit_reason"],
    )

    return row