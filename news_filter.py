# news_filter.py
"""
news_filter.py

Provides is_news_blackout() for the EUR/USD mean-reversion bot.

Rule: block trading signals if any HIGH-impact economic event is
scheduled within the next 2 hours of current_time (inclusive on both ends).
Past events (even high-impact) do NOT block.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

BLACKOUT_WINDOW = timedelta(hours=2)
HIGH_IMPACT = "high"


def _parse_event_timestamp(raw: object, event_index: int) -> datetime:
    """
    Accept a tz-aware datetime or an ISO-8601 string with timezone offset.
    Return a tz-aware datetime (timezone preserved, not converted to UTC).
    Raise ValueError for anything else or for naive datetimes.
    """
    if isinstance(raw, datetime):
        if raw.tzinfo is None or raw.utcoffset() is None:
            raise ValueError(
                f"Event[{event_index}] 'timestamp' is a naive datetime; "
                "tz-aware datetime required."
            )
        return raw

    if isinstance(raw, str):
        try:
            normalised = raw.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalised)
        except ValueError:
            raise ValueError(
                f"Event[{event_index}] 'timestamp' string '{raw}' is not a "
                "valid ISO-8601 datetime."
            )
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError(
                f"Event[{event_index}] 'timestamp' string '{raw}' has no "
                "timezone offset; tz-aware value required."
            )
        return dt

    raise ValueError(
        f"Event[{event_index}] 'timestamp' must be a tz-aware datetime or "
        f"an ISO-8601 string, got {type(raw).__name__!r}."
    )


def is_news_blackout(current_time: datetime, events: list) -> bool:
    """
    Return True if any HIGH-impact economic event falls within the window
    [current_time, current_time + 2h] (inclusive on both ends).
    Past events (event_dt < current_time) do NOT block.

    Parameters
    ----------
    current_time : datetime
        Must be tz-aware.
    events : list[dict]
        Each dict must contain:
          - 'timestamp': tz-aware datetime or ISO-8601 string with offset
          - 'impact':    str, one of 'high', 'medium', 'low' (case-insensitive)
        Optional:
          - 'name': str (used only in error messages)

    Returns
    -------
    bool
        True  if at least one high-impact event is in [now, now + 2h]
        False otherwise

    Raises
    ------
    ValueError
        - current_time is naive or not a datetime
        - events is not a list
        - any event is missing required keys or has bad values
        - any event timestamp is naive or unparseable
    """
    if not isinstance(current_time, datetime):
        raise ValueError(
            f"current_time must be a datetime, got {type(current_time).__name__!r}."
        )
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError(
            "current_time is a naive datetime; tz-aware datetime required."
        )

    if not isinstance(events, list):
        raise ValueError(
            f"events must be a list, got {type(events).__name__!r}."
        )

    window_end    = current_time + BLACKOUT_WINDOW
    valid_impacts = {"high", "medium", "low"}

    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(
                f"Event[{idx}] must be a dict, got {type(event).__name__!r}."
            )

        if "timestamp" not in event:
            raise ValueError(f"Event[{idx}] is missing required key 'timestamp'.")

        if "impact" not in event:
            raise ValueError(f"Event[{idx}] is missing required key 'impact'.")

        impact_raw = event["impact"]
        if not isinstance(impact_raw, str):
            raise ValueError(
                f"Event[{idx}] 'impact' must be a string, "
                f"got {type(impact_raw).__name__!r}."
            )
        impact = impact_raw.strip().lower()
        if impact not in valid_impacts:
            raise ValueError(
                f"Event[{idx}] 'impact' is {impact_raw!r}; "
                f"must be one of {sorted(valid_impacts)}."
            )

        if impact != HIGH_IMPACT:
            continue

        event_dt = _parse_event_timestamp(event["timestamp"], idx)

        # Past events do not block
        if event_dt < current_time:
            continue

        # Inclusive window: [current_time, current_time + 2h]
        if current_time <= event_dt <= window_end:
            return True

    return False


if __name__ == "__main__":
    from datetime import timezone

    now = datetime(2024, 6, 3, 14, 0, tzinfo=timezone.utc)

    events = [
        {"timestamp": "2024-06-03T14:30:00+00:00", "impact": "high",   "name": "NFP"},
        {"timestamp": "2024-06-03T17:00:00+00:00", "impact": "high",   "name": "FOMC"},
        {"timestamp": "2024-06-03T14:30:00+00:00", "impact": "medium", "name": "ISM"},
        {"timestamp": "2024-06-03T13:50:00Z",       "impact": "high",   "name": "CPI (past)"},
    ]

    assert is_news_blackout(now, events) is True
    print("Test 1 passed: blocked by upcoming NFP")

    no_block = [e for e in events if e["name"] != "NFP"]
    assert is_news_blackout(now, no_block) is False
    print("Test 2 passed: no block when NFP removed")

    assert is_news_blackout(now, []) is False
    print("Test 3 passed: empty list")

    # Exactly at window_end (inclusive)
    at_boundary = [{"timestamp": datetime(2024, 6, 3, 16, 0, tzinfo=timezone.utc),
                    "impact": "high"}]
    assert is_news_blackout(now, at_boundary) is True
    print("Test 4 passed: event exactly at 2h boundary is blocked (inclusive)")

    # 1 second past window_end (not blocked)
    past_boundary = [{"timestamp": datetime(2024, 6, 3, 16, 0, 1, tzinfo=timezone.utc),
                      "impact": "high"}]
    assert is_news_blackout(now, past_boundary) is False
    print("Test 5 passed: event 1 second past 2h boundary is not blocked")

    print("\nAll smoke tests passed.")
