# executor.py
"""
executor.py
-----------
OANDA v20 order execution for the EUR/USD mean-reversion bot.

Public API
----------
execute_signal(signal: dict, use_practice: bool = True) -> str
    Places a market order with attached stop-loss and take-profit.
    Returns the OANDA trade ID as a string.

Credentials
-----------
Fetched from AWS Secrets Manager, secret name: trading-bot/oanda
Required keys: oanda_api_token, oanda_account_id

Position sizing
---------------
For EUR/USD the P&L in USD is: (exit_price - entry_price) * units
Maximum loss at stop: abs(expected_entry - stop_price) * units = risk_usd
Therefore: units = floor(account_nav * RISK_FRACTION / stop_distance)

expected_entry is fetched from OANDA live pricing immediately before
order placement (ask for longs, bid for shorts), NOT the stale signal
close price. This ensures actual risk never exceeds 3% of NAV.

A MAX_ENTRY_DEVIATION check refuses execution if the live price has
drifted more than MAX_ENTRY_DEVIATION pips from the signal price,
which would invalidate the trade's risk/reward profile.

Account hard stop
-----------------
If account NAV is at or below ACCOUNT_HARD_STOP_USD ($120), all trading
is paused and ExecutionError is raised before any order is placed.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

PRACTICE_URL          = "https://api-fxpractice.oanda.com"
LIVE_URL              = "https://api-fxtrade.oanda.com"
INSTRUMENT            = "EUR_USD"
RISK_FRACTION         = 0.03    # 3% of account NAV per trade
ACCOUNT_HARD_STOP_USD = 120.0   # pause trading if NAV reaches this level
MAX_ENTRY_DEVIATION   = 0.0020  # 20 pip max drift from signal price (price units)
SECRET_NAME           = "trading-bot/oanda"
REQUEST_TIMEOUT       = 10      # seconds


# ── exceptions ───────────────────────────────────────────────────────────────

class ExecutionError(Exception):
    """Raised on any execution failure: API errors, validation, sizing, etc."""


# ── OANDA error parsing ───────────────────────────────────────────────────────

def _extract_oanda_error(resp: requests.Response) -> str:
    """
    Extract the most informative error message from an OANDA error response.
    Prefers structured fields; falls back to raw text.
    """
    try:
        body = resp.json()
    except Exception:
        return resp.text or f"HTTP {resp.status_code} (no body)"

    for path in (
        ("errorMessage",),
        ("orderRejectTransaction", "rejectReason"),
        ("orderCancelTransaction", "reason"),
    ):
        node = body
        for key in path:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                node = None
                break
        if node:
            return str(node)

    return resp.text or f"HTTP {resp.status_code}"


# ── credential loading ────────────────────────────────────────────────────────

def _load_credentials(region_name: str = "us-east-1") -> tuple[str, str]:
    """
    Fetch oanda_api_token and oanda_account_id from AWS Secrets Manager.

    Returns
    -------
    (api_token, account_id) — both non-empty stripped strings.

    Raises
    ------
    ExecutionError on any AWS or parsing failure.
    """
    client = boto3.client("secretsmanager", region_name=region_name)
    try:
        response = client.get_secret_value(SecretId=SECRET_NAME)
    except ClientError as exc:
        raise ExecutionError(
            f"AWS Secrets Manager error fetching '{SECRET_NAME}': {exc}"
        ) from exc
    except BotoCoreError as exc:
        raise ExecutionError(
            f"AWS BotoCore error fetching '{SECRET_NAME}': {exc}"
        ) from exc

    raw = response.get("SecretString")
    if raw is None:
        raise ExecutionError(
            f"Secret '{SECRET_NAME}' has no SecretString "
            "(binary secrets not supported)."
        )

    try:
        secret = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExecutionError(
            f"Secret '{SECRET_NAME}' is not valid JSON: {exc}"
        ) from exc

    for key in ("oanda_api_token", "oanda_account_id"):
        if key not in secret:
            raise ExecutionError(
                f"Secret '{SECRET_NAME}' is missing required key '{key}'."
            )
        if not isinstance(secret[key], str) or not secret[key].strip():
            raise ExecutionError(
                f"Secret '{SECRET_NAME}' key '{key}' must be a non-empty string."
            )

    return secret["oanda_api_token"].strip(), secret["oanda_account_id"].strip()


# ── OANDA helpers ─────────────────────────────────────────────────────────────

def _headers(api_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_token}",
        "Content-Type":  "application/json",
    }


def _get_account_nav(
    base_url: str,
    api_token: str,
    account_id: str,
) -> float:
    """
    Fetch account NAV (net asset value / equity) from OANDA.

    NAV is used for risk sizing because it reflects unrealised P&L and
    gives a conservative risk picture when positions are already open.

    Returns
    -------
    float  Account NAV in account currency (USD).

    Raises
    ------
    ExecutionError on HTTP or parsing failure.
    """
    url = f"{base_url}/v3/accounts/{account_id}/summary"
    try:
        resp = requests.get(
            url, headers=_headers(api_token), timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        raise ExecutionError(
            f"Network error fetching account summary: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise ExecutionError(
            f"OANDA account summary returned HTTP {resp.status_code}: "
            f"{_extract_oanda_error(resp)}"
        )

    try:
        nav = float(resp.json()["account"]["NAV"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ExecutionError(
            f"Could not parse account NAV from response: {exc}\n{resp.text}"
        ) from exc

    return nav


def _get_live_price(
    base_url: str,
    api_token: str,
    account_id: str,
    instrument: str = INSTRUMENT,
) -> tuple[float, float]:
    """
    Fetch current tradeable ask and bid for *instrument* from OANDA pricing.

    Uses /v3/accounts/{account_id}/pricing which returns account-specific
    prices including any applicable spread markups.

    Returns
    -------
    (ask, bid) as floats.

    Raises
    ------
    ExecutionError on HTTP or parsing failure.
    """
    url = f"{base_url}/v3/accounts/{account_id}/pricing"
    params = {"instruments": instrument}
    try:
        resp = requests.get(
            url,
            headers=_headers(api_token),
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ExecutionError(
            f"Network error fetching live price for {instrument}: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise ExecutionError(
            f"OANDA pricing endpoint returned HTTP {resp.status_code}: "
            f"{_extract_oanda_error(resp)}"
        )

    try:
        prices = resp.json()["prices"]
        if not prices:
            raise ExecutionError(
                f"OANDA pricing returned empty prices list for {instrument}."
            )
        price_obj = prices[0]
        ask = float(price_obj["asks"][0]["price"])
        bid = float(price_obj["bids"][0]["price"])
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ExecutionError(
            f"Could not parse live price from pricing response: {exc}\n{resp.text}"
        ) from exc

    if ask <= 0 or bid <= 0 or ask < bid:
        raise ExecutionError(
            f"Implausible live prices for {instrument}: ask={ask}, bid={bid}."
        )

    return ask, bid


def _has_open_position(
    base_url: str,
    api_token: str,
    account_id: str,
    instrument: str = INSTRUMENT,
) -> bool:
    """
    Return True if there is a non-zero open position for *instrument*.

    OANDA behaviour:
    - 404: no position resource exists for this instrument → no position.
    - 200: position object returned; non-zero long or short units → open.
      Zero units on both sides means no effective position.

    Raises
    ------
    ExecutionError on unexpected HTTP errors.
    """
    url = f"{base_url}/v3/accounts/{account_id}/positions/{instrument}"
    try:
        resp = requests.get(
            url, headers=_headers(api_token), timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        raise ExecutionError(
            f"Network error checking open position for {instrument}: {exc}"
        ) from exc

    if resp.status_code == 404:
        return False

    if resp.status_code != 200:
        raise ExecutionError(
            f"OANDA positions endpoint returned HTTP {resp.status_code}: "
            f"{_extract_oanda_error(resp)}"
        )

    try:
        pos = resp.json()["position"]
        long_units  = int(float(pos["long"]["units"]))
        short_units = int(float(pos["short"]["units"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise ExecutionError(
            f"Could not parse position units from response: {exc}\n{resp.text}"
        ) from exc

    return long_units != 0 or short_units != 0


def _calc_units(
    nav: float,
    expected_entry: float,
    stop_price: float,
    side: str,
) -> int:
    """
    Calculate position size so that the maximum loss equals 3% of NAV.

    EUR/USD P&L formula (USD):
        p&l = (exit_price - entry_price) * units

    Maximum loss at stop:
        loss = abs(expected_entry - stop_price) * units = nav * RISK_FRACTION

    Therefore:
        units = floor(nav * RISK_FRACTION / abs(expected_entry - stop_price))

    Parameters
    ----------
    expected_entry : float
        Live ask (long) or bid (short) fetched immediately before order
        placement — NOT the stale signal close price.

    Returns a positive int for longs, negative int for shorts
    (OANDA convention: positive units = buy, negative units = sell).

    Raises
    ------
    ExecutionError if stop_distance is zero or units rounds to 0.
    """
    stop_distance = abs(expected_entry - stop_price)
    if stop_distance == 0.0:
        raise ExecutionError(
            "stop_distance is zero (expected_entry == stop_price) — "
            "cannot size position."
        )

    risk_usd = nav * RISK_FRACTION
    units    = math.floor(risk_usd / stop_distance)

    if units < 1:
        raise ExecutionError(
            f"Calculated units={units} (nav={nav:.2f}, "
            f"risk_usd={risk_usd:.2f}, stop_distance={stop_distance:.5f}). "
            "Position too small to place — account NAV may be too low."
        )

    return units if side == "long" else -units


def _place_order(
    base_url: str,
    api_token: str,
    account_id: str,
    instrument: str,
    units: int,
    stop_price: float,
    target_price: float,
) -> str:
    """
    Place a market FOK order with attached stop-loss and take-profit.

    Handles the three fill-transaction variants OANDA may return:
    - tradeOpened  : expected case (new position opened) → return trade ID
    - tradesClosed : unexpected given pre-check → raise ExecutionError
    - tradeReduced : unexpected given pre-check → raise ExecutionError

    Returns
    -------
    str  OANDA trade ID of the newly opened trade.

    Raises
    ------
    ExecutionError on HTTP errors, order rejection, or unexpected fill type.
    """
    body: dict[str, Any] = {
        "order": {
            "type":         "MARKET",
            "instrument":   instrument,
            "units":        str(units),
            "timeInForce":  "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {
                "price":       f"{stop_price:.5f}",
                "timeInForce": "GTC",
            },
            "takeProfitOnFill": {
                "price":       f"{target_price:.5f}",
                "timeInForce": "GTC",
            },
        }
    }

    url = f"{base_url}/v3/accounts/{account_id}/orders"
    try:
        resp = requests.post(
            url,
            headers=_headers(api_token),
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ExecutionError(f"Network error placing order: {exc}") from exc

    if resp.status_code not in (200, 201):
        raise ExecutionError(
            f"OANDA order endpoint returned HTTP {resp.status_code}: "
            f"{_extract_oanda_error(resp)}"
        )

    data = resp.json()
    fill = data.get("orderFillTransaction", {})

    # Expected: fresh trade opened
    trade_opened = fill.get("tradeOpened")
    if trade_opened and trade_opened.get("tradeID"):
        trade_id = str(trade_opened["tradeID"])
        logger.info(
            "Order filled | instrument=%s units=%d stop=%.5f target=%.5f "
            "trade_id=%s",
            instrument, units, stop_price, target_price, trade_id,
        )
        return trade_id

    # Unexpected: order closed an existing position (race condition)
    if fill.get("tradesClosed"):
        raise ExecutionError(
            "Order unexpectedly closed an existing position. "
            "A race condition may have allowed a concurrent position to open. "
            f"Fill transaction: {fill}"
        )

    # Unexpected: order reduced an existing position
    if fill.get("tradeReduced"):
        raise ExecutionError(
            "Order unexpectedly reduced an existing position. "
            f"Fill transaction: {fill}"
        )

    # Order was not filled (FOK cancelled or rejected)
    raise ExecutionError(
        f"Order was not filled. OANDA response: {_extract_oanda_error(resp)}"
    )


# ── public API ────────────────────────────────────────────────────────────────

def execute_signal(
    signal: dict,
    use_practice: bool = True,
    aws_region: str = "us-east-1",
) -> str:
    """
    Execute a trading signal produced by evaluate_signal().

    Steps
    -----
    1. Validate the signal dict and price relationships.
    2. Load credentials from AWS Secrets Manager.
    3. Select practice or live endpoint.
    4. Fetch account NAV; enforce hard stop ($120 floor).
    5. Check for an existing open EUR_USD position; raise if one exists.
    6. Fetch live ask/bid from OANDA pricing.
    7. Validate live price has not drifted beyond MAX_ENTRY_DEVIATION (20 pips)
       from signal price; raise if stale.
    8. Size units using live expected_entry and stop_price.
    9. Place market order with stop-loss and take-profit attached.
    10. Return the OANDA trade ID.

    Parameters
    ----------
    signal : dict
        Must contain keys produced by evaluate_signal():
        'signal'        : 'long' or 'short'
        'entry_price'   : float, last closed 1h candle close (signal price)
        'stop_price'    : float, stop-loss level
        'target_price'  : float, take-profit level (daily open)

        Price relationships enforced:
        long  → stop_price < entry_price < target_price
        short → target_price < entry_price < stop_price

    use_practice : bool
        True  (default) → https://api-fxpractice.oanda.com
        False           → https://api-fxtrade.oanda.com

    aws_region : str
        AWS region where the secret is stored (default 'us-east-1').

    Returns
    -------
    str
        OANDA trade ID of the opened position.

    Raises
    ------
    ValueError      : signal dict is malformed or prices are inconsistent.
    ExecutionError  : credential errors, hard stop triggered, existing
                      position, live price drift exceeded, API errors,
                      sizing errors, or fill failures.
    """
    # -- validate signal type --
    if not isinstance(signal, dict):
        raise ValueError(
            f"signal must be a dict, got {type(signal).__name__!r}."
        )

    required_keys = {"signal", "entry_price", "stop_price", "target_price"}
    missing = required_keys - set(signal.keys())
    if missing:
        raise ValueError(f"signal dict is missing keys: {sorted(missing)}")

    side = signal["signal"]
    if side not in ("long", "short"):
        raise ValueError(
            f"signal['signal'] must be 'long' or 'short', got {side!r}."
        )

    try:
        signal_entry = float(signal["entry_price"])
        stop_price   = float(signal["stop_price"])
        target_price = float(signal["target_price"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"signal prices must be numeric: {exc}") from exc

    for name, val in (("entry_price", signal_entry),
                      ("stop_price",  stop_price),
                      ("target_price", target_price)):
        if not math.isfinite(val) or val <= 0:
            raise ValueError(
                f"signal['{name}'] must be a finite positive number, "
                f"got {val!r}."
            )

    # -- validate price relationships using signal entry --
    if side == "long":
        if not (stop_price < signal_entry < target_price):
            raise ValueError(
                "Long signal requires stop_price < entry_price < target_price. "
                f"Got stop={stop_price:.5f}, entry={signal_entry:.5f}, "
                f"target={target_price:.5f}."
            )
    else:
        if not (target_price < signal_entry < stop_price):
            raise ValueError(
                "Short signal requires target_price < entry_price < stop_price. "
                f"Got target={target_price:.5f}, entry={signal_entry:.5f}, "
                f"stop={stop_price:.5f}."
            )

    # -- load credentials --
    api_token, account_id = _load_credentials(region_name=aws_region)

    base_url = PRACTICE_URL if use_practice else LIVE_URL
    logger.info(
        "Executor using %s endpoint.", "practice" if use_practice else "LIVE"
    )

    # -- fetch NAV --
    nav = _get_account_nav(base_url, api_token, account_id)
    logger.info("Account NAV: %.2f USD", nav)

    # -- account hard stop --
    if nav <= ACCOUNT_HARD_STOP_USD:
        raise ExecutionError(
            f"Account NAV {nav:.2f} USD is at or below the hard stop of "
            f"{ACCOUNT_HARD_STOP_USD:.2f} USD. All trading paused."
        )

    # -- guard: no existing position --
    if _has_open_position(base_url, api_token, account_id):
        raise ExecutionError(
            f"An open {INSTRUMENT} position already exists. "
            "Close it before placing a new signal."
        )

    # -- fetch live price --
    ask, bid = _get_live_price(base_url, api_token, account_id)
    expected_entry = ask if side == "long" else bid
    logger.info(
        "Live price | ask=%.5f bid=%.5f | using expected_entry=%.5f for %s",
        ask, bid, expected_entry, side,
    )

    # -- drift guard --
    drift = abs(expected_entry - signal_entry)
    if drift > MAX_ENTRY_DEVIATION:
        raise ExecutionError(
            f"Live price drift of {drift:.5f} ({drift * 10000:.1f} pips) exceeds "
            f"maximum allowed {MAX_ENTRY_DEVIATION:.5f} "
            f"({MAX_ENTRY_DEVIATION * 10000:.1f} pips). "
            f"Signal entry={signal_entry:.5f}, live expected_entry={expected_entry:.5f}. "
            "Signal may be stale — execution refused."
        )

    # -- size position using live expected_entry --
    units = _calc_units(nav, expected_entry, stop_price, side)
    logger.info(
        "Signal: %s | signal_entry=%.5f expected_entry=%.5f "
        "stop=%.5f target=%.5f | units=%d",
        side, signal_entry, expected_entry, stop_price, target_price, units,
    )

    # -- place order --
    trade_id = _place_order(
        base_url, api_token, account_id,
        instrument=INSTRUMENT,
        units=units,
        stop_price=stop_price,
        target_price=target_price,
    )

    return trade_id