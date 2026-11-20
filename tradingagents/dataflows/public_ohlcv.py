"""Public A-share daily OHLCV parsing.

Tencent's daily K-line (``web.ifzq.gtimg.cn/.../fqkline/get``) is the default
source. Each bar is ``[date, open, close, high, low, volume]`` — close is the
second price, not the third. Volume is in 手 (1 lot = 100 shares). Missing
prices and volume are dropped, never filled with 0. Bars whose Shanghai
trading date is after the requested cutoff are dropped, never rewritten.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

# Asia/Shanghai has no daylight-saving time. A fixed UTC+8 offset matches the
# market clock used by ``_MARKET_TZ`` without depending on the host timezone.
SHANGHAI_TZ = timezone(timedelta(hours=8))

OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

TENCENT_KLINE_ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
SINA_KLINE_ENDPOINT = (
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UNIX_SEC = re.compile(r"^\d{10}(?:\.\d+)?$")
_UNIX_MS = re.compile(r"^\d{13}(?:\.\d+)?$")

_BLOCK_MARKERS = ("hit risk", "iscaptchaenabled", "captcha", "验证码")

VOLUME_LABELS = {
    "tencent": (
        "手 (1手=100股). Tencent daily bars do not include turnover amount. "
        "Values are not converted to shares, and missing volume is not filled with 0."
    ),
    "sina": (
        "Sina CN_MarketData.getKLineData native volume. The unit is not certified "
        "and is not labeled as 手. Values are not converted, and missing volume "
        "is not filled with 0."
    ),
    "mootdx": (
        "mootdx native volume (not converted, not labeled as 手). "
        "Missing volume is not filled with 0."
    ),
}

ADJUST_LABELS = {
    "none": "none (不复权, unadjusted)",
    "qfq": "qfq (前复权, forward-adjusted)",
    "unspecified": "not declared by this source (not Tencent none/qfq)",
}


class OhlcvFetchError(ValueError):
    """The OHLCV request cannot be fulfilled. No prices were invented."""

    def __init__(
        self,
        message: str,
        *,
        source: str,
        status: int | None = None,
        body_len: int | None = None,
    ):
        super().__init__(message)
        self.source = source
        self.status = status
        self.body_len = body_len


def response_is_blocked(status: int | None, text: str | None) -> bool:
    """True for an explicit refusal. Callers must not retry or rotate exits."""
    if status in (401, 403, 429, 451):
        return True
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _BLOCK_MARKERS)


def tencent_kline_param(symbol: str, end_date: str, adjust: str, count: int) -> str:
    """Query param. ``adjust`` is ``none`` or ``qfq``; none sends an empty slot."""
    adj = "" if adjust == "none" else adjust
    return f"{symbol},day,,{end_date},{int(count)},{adj}"


def shanghai_calendar_date(value) -> pd.Timestamp | None:
    """Trading-day label in Asia/Shanghai, or None when the value is unusable.

    A bare ``YYYY-MM-DD`` is that calendar date. It is not parsed as UTC
    midnight. Unix instants and timezone-aware values are converted to the
    Shanghai calendar date. Naive datetimes that include a time are Shanghai
    wall time, matching the news filter.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "nan", "nat"}:
            return None
        if _DATE_ONLY.match(text):
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d")
            except ValueError:
                return None
            return pd.Timestamp(parsed.date())
        if _UNIX_SEC.match(text) or _UNIX_MS.match(text):
            value = float(text)
        else:
            iso = text.replace("Z", "+00:00")
            if " " in iso and "T" not in iso:
                iso = iso.replace(" ", "T", 1)
            try:
                parsed = datetime.fromisoformat(iso)
            except ValueError:
                return None
            return _timestamp_from_datetime(parsed)

    if isinstance(value, datetime):
        return _timestamp_from_datetime(value)
    if isinstance(value, (int, float)):
        return _timestamp_from_unix(float(value))
    return None


def _timestamp_from_datetime(parsed: datetime) -> pd.Timestamp:
    if parsed.tzinfo is None:
        aware = parsed.replace(tzinfo=SHANGHAI_TZ)
    else:
        aware = parsed.astimezone(SHANGHAI_TZ)
    return pd.Timestamp(year=aware.year, month=aware.month, day=aware.day)


def _timestamp_from_unix(number: float) -> pd.Timestamp | None:
    if not math.isfinite(number) or number < 0:
        return None
    if 1_000_000_000 <= number < 100_000_000_000:
        seconds = number
    elif 1_000_000_000_000 <= number < 100_000_000_000_000:
        seconds = number / 1000.0
    else:
        return None
    try:
        aware = datetime.fromtimestamp(seconds, tz=SHANGHAI_TZ)
    except (OverflowError, OSError, ValueError):
        return None
    return pd.Timestamp(year=aware.year, month=aware.month, day=aware.day)


def _finite_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "nan", "nat"}:
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clean_ohlcv_rows(
    rows: list[dict],
    *,
    end_date: str,
    adjust: str,
    source: str,
) -> tuple[pd.DataFrame, dict]:
    """Drop future and unusable bars. Never insert a bar or fill a gap with 0.

    ``adjust="none"`` drops non-positive prices. ``adjust="qfq"`` keeps
    non-positive forward-adjusted prices and counts them, because a long
    qfq history can print negative prices after a large dividend. Those
    values are the vendor's numbers, not a zero fill.
    """
    cutoff = shanghai_calendar_date(end_date)
    if cutoff is None:
        raise OhlcvFetchError(
            "OHLCV request cannot be fulfilled: "
            f"cutoff {end_date!r} is not a YYYY-MM-DD trading date. "
            "No prices were invented.",
            source=source,
        )

    kept: list[dict] = []
    dropped_future = 0
    dropped_invalid = 0
    nonpositive_kept = 0
    for row in rows:
        trading_day = shanghai_calendar_date(row.get("date"))
        open_ = _finite_number(row.get("open"))
        high = _finite_number(row.get("high"))
        low = _finite_number(row.get("low"))
        close = _finite_number(row.get("close"))
        volume = _finite_number(row.get("volume"))
        if (
            trading_day is None
            or open_ is None
            or high is None
            or low is None
            or close is None
            or volume is None
            or volume < 0
            or high < low
        ):
            dropped_invalid += 1
            continue
        if trading_day > cutoff:
            dropped_future += 1
            continue
        prices = (open_, high, low, close)
        if adjust == "none" and any(price <= 0 for price in prices):
            dropped_invalid += 1
            continue
        if adjust == "qfq" and any(price <= 0 for price in prices):
            nonpositive_kept += 1
        kept.append(
            {
                "Date": trading_day,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )

    frame = pd.DataFrame(kept, columns=OHLCV_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["Date"], keep="last")
        frame = frame.sort_values("Date").reset_index(drop=True)

    first = None if frame.empty else frame["Date"].min().strftime("%Y-%m-%d")
    last = None if frame.empty else frame["Date"].max().strftime("%Y-%m-%d")
    meta = {
        "schema": 1,
        "source": source,
        "adjust": adjust,
        "adjust_label": ADJUST_LABELS.get(adjust, adjust),
        "volume_unit_label": VOLUME_LABELS.get(source, VOLUME_LABELS["mootdx"]),
        "asof_cutoff": cutoff.strftime("%Y-%m-%d"),
        "asof_timezone": "Asia/Shanghai",
        "bars_after_cutoff_dropped": dropped_future,
        "rows_dropped_invalid": dropped_invalid,
        "nonpositive_price_kept": nonpositive_kept,
        "row_count": int(len(frame)),
        "first_date": first,
        "last_date": last,
        "from_cache": False,
    }
    return frame, meta


def parse_tencent_daily_payload(
    payload,
    *,
    symbol: str,
    adjust: str,
    end_date: str,
    param: str,
) -> tuple[pd.DataFrame, dict]:
    """Parse one fqkline payload. Refuses to swap ``day`` and ``qfqday``."""
    if not isinstance(payload, dict):
        raise OhlcvFetchError(
            "OHLCV request cannot be fulfilled: Tencent payload is not a JSON object. "
            f"symbol={symbol} adjust={adjust} cutoff={end_date} Asia/Shanghai. "
            "No prices were invented and missing bars were not filled with zeros.",
            source="tencent",
        )
    code = payload.get("code")
    if code not in (0, "0", None):
        raise OhlcvFetchError(
            "OHLCV request cannot be fulfilled: "
            f"Tencent code={code!r} msg={payload.get('msg')!r}. "
            f"symbol={symbol} adjust={adjust} cutoff={end_date} Asia/Shanghai. "
            "No prices were invented and missing bars were not filled with zeros.",
            source="tencent",
        )
    data = payload.get("data")
    if not isinstance(data, dict) or symbol not in data or not isinstance(data.get(symbol), dict):
        keys = list(data)[:8] if isinstance(data, dict) else []
        raise OhlcvFetchError(
            "OHLCV request cannot be fulfilled: "
            f"Tencent payload has no daily node for {symbol}. keys={keys}. "
            f"adjust={adjust} cutoff={end_date} Asia/Shanghai. "
            "No prices were invented and missing bars were not filled with zeros.",
            source="tencent",
        )
    node = data[symbol]
    series_key = "day" if adjust == "none" else "qfqday"
    series = node.get(series_key)
    if not isinstance(series, list):
        raise OhlcvFetchError(
            "OHLCV request cannot be fulfilled: "
            f"Tencent response has no {series_key!r} bars for {symbol} "
            f"(adjust={adjust}). keys={list(node)[:12]}. "
            "A different adjustment series was not substituted. "
            "No prices were invented and missing bars were not filled with zeros.",
            source="tencent",
        )

    mapped = []
    for raw in series:
        if not isinstance(raw, (list, tuple)) or len(raw) < 6:
            mapped.append(
                {"date": None, "open": None, "high": None, "low": None, "close": None, "volume": None}
            )
            continue
        # Vendor order is date, open, close, high, low, volume. Not OHLC.
        mapped.append(
            {
                "date": raw[0],
                "open": raw[1],
                "close": raw[2],
                "high": raw[3],
                "low": raw[4],
                "volume": raw[5],
            }
        )
    frame, meta = clean_ohlcv_rows(mapped, end_date=end_date, adjust=adjust, source="tencent")
    meta["endpoint"] = TENCENT_KLINE_ENDPOINT
    meta["request_param"] = param
    meta["series_key"] = series_key
    if frame.empty:
        raise OhlcvFetchError(
            "OHLCV request cannot be fulfilled: "
            f"Tencent returned no usable daily bars for {symbol}. "
            f"adjust={adjust} cutoff={end_date} Asia/Shanghai. "
            f"bars_after_cutoff_dropped={meta['bars_after_cutoff_dropped']} "
            f"rows_dropped_invalid={meta['rows_dropped_invalid']}. "
            "No prices were invented and missing bars were not filled with zeros.",
            source="tencent",
        )
    return frame, meta


def format_ohlcv_provenance(meta: dict | None, *, fallback_cutoff: str = "") -> str:
    """Comment block stating source, adjustment, volume unit, and cutoff."""
    meta = meta or {}
    if not meta:
        cutoff = fallback_cutoff or "(unknown)"
        return (
            "# Data source: mootdx (TCP 7709 bars), legacy OHLCV path\n"
            "# Adjustment: not declared by this source (not Tencent none/qfq)\n"
            f"# Volume unit: {VOLUME_LABELS['mootdx']}\n"
            f"# As-of cutoff: {cutoff} inclusive (Asia/Shanghai trading date). "
            "Bars after this date are excluded and not filled.\n"
        )

    lines = [
        f"# Data source: {meta.get('source', 'unknown')}"
        + (f" ({meta['endpoint']})" if meta.get("endpoint") else ""),
        f"# Adjustment: {meta.get('adjust_label', meta.get('adjust', 'unspecified'))}",
        f"# Volume unit: {meta.get('volume_unit_label', VOLUME_LABELS['mootdx'])}",
        (
            f"# As-of cutoff: {meta.get('asof_cutoff', fallback_cutoff)} inclusive "
            f"({meta.get('asof_timezone', 'Asia/Shanghai')} trading date). "
            "Bars after this date are dropped and not filled."
        ),
        f"# Bars after cutoff dropped: {meta.get('bars_after_cutoff_dropped', 0)}",
        (
            "# Invalid bars dropped (missing, non-finite, or inconsistent OHLC/volume; "
            f"not replaced with 0): {meta.get('rows_dropped_invalid', 0)}"
        ),
        (
            f"# Loaded bars on or before cutoff: {meta.get('row_count', '')} "
            f"({meta.get('first_date')} .. {meta.get('last_date')})"
        ),
    ]
    if meta.get("request_param"):
        lines.append(f"# Request param: {meta['request_param']}")
    if meta.get("series_key"):
        lines.append(f"# Response series: {meta['series_key']}")
    if meta.get("from_cache"):
        lines.append(
            "# Cache: reused the stored series for this source, adjustment, and cutoff."
        )
    kept = meta.get("nonpositive_price_kept") or 0
    if kept:
        lines.append(
            f"# Warning: {kept} forward-adjusted bars contain a non-positive price. "
            "Those vendor values were kept and were not replaced with 0."
        )
    lines.append(
        "# Missing sessions are left missing. No bar was synthesized and no price was filled with 0."
    )
    return "\n".join(lines) + "\n"
