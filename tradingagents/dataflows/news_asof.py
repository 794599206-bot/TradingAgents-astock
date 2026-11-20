"""Strict Asia/Shanghai as-of filter for A-share news items.

Every branch (stock news, CLS wire, Eastmoney 7x24) must pass through
``filter_news_items`` before it can be shown to a model. Items with a missing
or unparseable timestamp are dropped and counted. Title text and URL dates
are not timestamps. Naive wall times are Shanghai time, not the host zone.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

# Same fixed offset as the market clock in a_stock._MARKET_TZ. China has no DST.
SHANGHAI_TZ = timezone(timedelta(hours=8))

_UNIX_SEC = re.compile(r"^\d{10}(?:\.\d+)?$")
_UNIX_MS = re.compile(r"^\d{13}(?:\.\d+)?$")
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_ONLY_CN = re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日$")
_BLANK_STRINGS = {"", "null", "none", "nan", "nat", "undefined", "n/a"}

_STRPTIME_FORMATS = (
    ("%Y-%m-%d %H:%M:%S", "datetime"),
    ("%Y-%m-%d %H:%M", "datetime"),
    ("%Y/%m/%d %H:%M:%S", "datetime"),
    ("%Y/%m/%d %H:%M", "datetime"),
    ("%Y-%m-%d", "date"),
    ("%Y/%m/%d", "date"),
    ("%Y年%m月%d日 %H:%M:%S", "datetime"),
    ("%Y年%m月%d日 %H:%M", "datetime"),
    ("%Y年%m月%d日", "date"),
)


@dataclass
class ParsedTime:
    instant: datetime
    precision: str  # "datetime" or "date"


@dataclass
class FilteredNews:
    kept: list[dict]
    dropped_after_asof: int = 0
    dropped_before_window: int = 0
    dropped_unparseable: int = 0
    dropped_duplicate: int = 0
    window_start: datetime | None = None
    asof: datetime | None = None
    unparseable_examples: list[str] = field(default_factory=list)


def format_shanghai(instant: datetime) -> str:
    if instant.tzinfo is None:
        aware = instant.replace(tzinfo=SHANGHAI_TZ)
    else:
        aware = instant.astimezone(SHANGHAI_TZ)
    text = aware.strftime("%Y-%m-%d %H:%M:%S")
    if aware.microsecond:
        text += f".{aware.microsecond:06d}"
    return text + " Asia/Shanghai"


def asof_bounds(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """Inclusive window: ``[start 00:00:00, end 23:59:59.999999]`` in Shanghai."""
    start = _require_day(start_date, "start_date")
    end = _require_day(end_date, "end_date")
    if start > end:
        raise ValueError(f"news window start {start_date} is after end {end_date}")
    window_start = datetime(start.year, start.month, start.day, tzinfo=SHANGHAI_TZ)
    next_midnight = datetime(end.year, end.month, end.day, tzinfo=SHANGHAI_TZ) + timedelta(days=1)
    asof = next_midnight - timedelta(microseconds=1)
    return window_start, asof


def _require_day(value: str, label: str) -> date:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD, got {value!r}") from exc


def parse_news_timestamp(value) -> ParsedTime | None:
    """Return a Shanghai instant, or None when the value cannot be a timestamp.

    Unix seconds and milliseconds are absolute instants. Timezone-aware
    strings are converted to Asia/Shanghai. Naive strings are Shanghai wall
    time. A calendar date alone is 00:00:00 on that Shanghai date.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return _attach(value, "datetime")
    if isinstance(value, date):
        return _attach(datetime(value.year, value.month, value.day), "date")
    if isinstance(value, (int, float)):
        return _from_unix(float(value))
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _BLANK_STRINGS:
            return None
        if _UNIX_SEC.match(text) or _UNIX_MS.match(text):
            return _from_unix(float(text))
        iso = text[:-1] + "+00:00" if text.endswith("Z") else text
        if " " in iso and "T" not in iso:
            iso_try = iso.replace(" ", "T", 1)
        else:
            iso_try = iso
        try:
            parsed = datetime.fromisoformat(iso_try)
        except ValueError:
            parsed = None
        if parsed is not None:
            precision = "date" if _DATE_ONLY.match(text) or _DATE_ONLY_CN.match(text) else "datetime"
            return _attach(parsed, precision)
        stripped = re.sub(r"\.\d+$", "", text)
        for fmt, precision in _STRPTIME_FORMATS:
            try:
                parsed = datetime.strptime(stripped, fmt)
            except ValueError:
                continue
            return _attach(parsed, precision)
        return None
    return None


def _attach(parsed: datetime, precision: str) -> ParsedTime:
    if parsed.tzinfo is None:
        aware = parsed.replace(tzinfo=SHANGHAI_TZ)
    else:
        aware = parsed.astimezone(SHANGHAI_TZ)
    return ParsedTime(aware, precision)


def _from_unix(number: float) -> ParsedTime | None:
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
    return ParsedTime(aware, "datetime")


def filter_news_items(items, start_date: str, end_date: str) -> FilteredNews:
    """Keep items at or before the as-of instant and inside the window.

    ``items`` entries use ``time``, ``title``, ``content``, ``source``, and
    ``url``. The title and URL are never parsed as a clock time.
    """
    window_start, asof = asof_bounds(start_date, end_date)
    result = FilteredNews(kept=[], window_start=window_start, asof=asof)
    pending: list[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            result.dropped_unparseable += 1
            continue
        parsed = parse_news_timestamp(item.get("time"))
        if parsed is None:
            result.dropped_unparseable += 1
            if len(result.unparseable_examples) < 3:
                result.unparseable_examples.append(repr(item.get("time"))[:80])
            continue
        if parsed.instant < window_start:
            result.dropped_before_window += 1
            continue
        if parsed.instant > asof:
            result.dropped_after_asof += 1
            continue
        pending.append(
            {
                "title": str(item.get("title") or "").strip() or "(no title)",
                "content": "" if item.get("content") is None else str(item.get("content")),
                "source": str(item.get("source") or "").strip() or "(source not labeled)",
                "url": "" if item.get("url") in (None, "nan") else str(item.get("url")),
                "raw_time": repr(item.get("time"))[:80],
                "published_at": parsed.instant,
                "precision": parsed.precision,
            }
        )

    seen: set[tuple] = set()
    for row in pending:
        key = (row["source"], row["title"], row["published_at"].isoformat(), row["url"])
        if key in seen:
            result.dropped_duplicate += 1
            continue
        seen.add(key)
        result.kept.append(row)
    result.kept.sort(key=lambda row: (row["published_at"], row["source"], row["title"]))
    return result


def render_news_report(
    heading: str,
    filtered: FilteredNews,
    *,
    sources_ok: list[str],
    fetch_errors: list[tuple[str, str]],
    shown_limit: int | None = None,
) -> str:
    """Render kept items with timestamps, sources, and drop counts.

    Exclusion is enforceable only when at least one source returned a list
    (possibly empty). A latest-page feed is still not a historical archive,
    so the report does not claim a complete historical forecast.
    """
    kept = filtered.kept
    omitted = 0
    shown = kept
    if shown_limit is not None:
        if shown_limit < 0:
            raise ValueError("news limit must be >= 0")
        if len(kept) > shown_limit:
            omitted = len(kept) - shown_limit
            shown = kept[-shown_limit:]

    lines = [heading.rstrip(), ""]
    if sources_ok:
        lines.append("# As-of exclusion filter: enforceable.")
        lines.append(
            "# Only items with a parseable timestamp at or before the as-of instant "
            "are listed. Unknown timestamps were dropped and were not treated as in-window."
        )
    else:
        lines.append("# As-of exclusion filter: NOT enforceable.")
        lines.append(
            "# No source returned a list of items, so timestamps could not be checked. "
            "Do not use this result as a historical news set or as forecast input."
        )
    if filtered.window_start is not None and filtered.asof is not None:
        lines.append(
            f"# Window: {format_shanghai(filtered.window_start)} <= published_at <= "
            f"{format_shanghai(filtered.asof)}"
        )
    lines.append(
        "# Naive timestamps are read as Asia/Shanghai wall time, not the host timezone and not UTC. "
        "Unix timestamps are absolute instants converted to Asia/Shanghai. "
        "Title text and URL dates are not timestamps."
    )
    if any(row["precision"] == "date" for row in shown):
        lines.append(
            "# Date-only timestamps are kept as 00:00:00 Asia/Shanghai on that calendar date. "
            "The minute of publication is unknown."
        )
    lines.append(
        "# Source limitation: these endpoints return a latest page, not a point-in-time archive. "
        "Items older than that page are absent. A missing item is not proof the market had no such news. "
        "Do not describe this output as a complete historical forecast."
    )
    lines.append(
        "# Dropped: "
        f"after_asof={filtered.dropped_after_asof} "
        f"before_window={filtered.dropped_before_window} "
        f"unparseable={filtered.dropped_unparseable} "
        f"duplicate={filtered.dropped_duplicate}"
    )
    if filtered.unparseable_examples:
        lines.append("# Unparseable examples: " + " | ".join(filtered.unparseable_examples))
    lines.append(f"# Sources returned: {', '.join(sources_ok) if sources_ok else '(none)'}")
    if fetch_errors:
        lines.append("# Fetch failures:")
        for name, detail in fetch_errors:
            lines.append(f"# - {name}: {detail}")
    lines.append(
        f"# Kept: {len(kept)} shown: {len(shown)} omitted_by_limit={omitted}"
    )
    lines.append("# Remaining timestamps and sources:")
    if not shown:
        lines.append("# - (none)")
        if sources_ok:
            lines.append(
                "# No listed item survived the filter. This is not evidence that no news was published."
            )
    else:
        for row in shown:
            lines.append(
                f"# - {format_shanghai(row['published_at'])} | {row['source']} | {row['title']}"
            )
    lines.append("")

    for row in shown:
        lines.append(f"### {row['title']} (source: {row['source']})")
        lines.append(f"- published_at: {format_shanghai(row['published_at'])}")
        lines.append(f"- timestamp_precision: {row['precision']}")
        lines.append(f"- source: {row['source']}")
        lines.append(f"- raw_time: {row['raw_time']}")
        content = row["content"]
        if content:
            snippet = content[:300] + "..." if len(content) > 300 else content
            lines.append(snippet)
        if row["url"]:
            lines.append(f"Link: {row['url']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
