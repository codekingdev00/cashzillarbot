"""Small time and formatting helpers. All timestamps are UTC milliseconds."""
from __future__ import annotations

import time
from datetime import datetime, timezone

_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}


def timeframe_to_ms(tf: str) -> int:
    tf = str(tf).strip()
    if len(tf) < 2 or tf[-1] not in _UNIT_MS or not tf[:-1].isdigit() or int(tf[:-1]) <= 0:
        raise ValueError(f"Unsupported timeframe '{tf}' (use e.g. 15m, 1h, 4h, 1d)")
    return int(tf[:-1]) * _UNIT_MS[tf[-1]]


def now_ms() -> int:
    return int(time.time() * 1000)


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def fmt_ts(ms: int) -> str:
    return ms_to_dt(ms).strftime("%Y-%m-%d %H:%M UTC")


def utc_day(ms: int) -> str:
    return ms_to_dt(ms).strftime("%Y-%m-%d")


def parse_utc(value: str) -> int:
    """Parse an ISO date/datetime (naive values are treated as UTC) to ms."""
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fmt_price(x: float | None) -> str:
    if x is None:
        return "-"
    mag = abs(x)
    if mag >= 1000:
        return f"{x:,.2f}"
    if mag >= 1:
        return f"{x:,.4f}"
    return f"{x:.6g}"


def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"
