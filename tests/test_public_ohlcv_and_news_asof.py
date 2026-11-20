"""OHLCV source provenance and strict news as-of filtering.

Covers Unix vs timezone-aware timestamps, the Shanghai day boundary, unknown
timestamps, bars after the cutoff, empty HTTP bodies, HTTP errors, volume
units, and the bars actually handed to the indicator calculator.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas as pd
import pytest

from tradingagents.dataflows import a_stock
from tradingagents.dataflows.news_asof import (
    SHANGHAI_TZ,
    filter_news_items,
    parse_news_timestamp,
)
from tradingagents.dataflows.public_ohlcv import (
    parse_tencent_daily_payload,
    shanghai_calendar_date,
)

SH = SHANGHAI_TZ
UTC = timezone.utc


class _Resp:
    def __init__(self, status, body, payload=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.status_code = status
        self.content = body
        self.text = body.decode("utf-8", errors="replace")
        self._payload = payload

    def json(self):
        if self._payload is not None:
            return self._payload
        return json.loads(self.content.decode("utf-8"))


def _tencent_row(date, open_, close, high, low, volume):
    """Tencent order is date, open, close, high, low, volume."""
    return [date, open_, close, high, low, volume]


def _use_tencent(monkeypatch, tmp_path, rows, *, adjust="none", status=200, body=None):
    calls = []
    symbol = "sh600519"
    series_key = "day" if adjust == "none" else "qfqday"

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if body is not None:
            return _Resp(status, body)
        payload = {"code": 0, "data": {symbol: {series_key: rows}}}
        return _Resp(200, json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(a_stock, "_http_get_public", fake_get)
    monkeypatch.setattr(a_stock, "_market_today", lambda: datetime(2026, 11, 20).date())
    monkeypatch.delenv("ASTOCK_OHLCV_SOURCE", raising=False)
    monkeypatch.delenv("ASTOCK_OHLCV_ADJUST", raising=False)
    monkeypatch.setenv("ASTOCK_OHLCV_SOURCE", "tencent")
    monkeypatch.setenv("ASTOCK_OHLCV_ADJUST", adjust)
    monkeypatch.setattr(
        "tradingagents.dataflows.config.get_config",
        lambda: {"data_cache_dir": str(tmp_path)},
    )
    return calls


def _csv_table(result: str) -> pd.DataFrame:
    _header, csv_text = result.split("\n\n", 1)
    return pd.read_csv(StringIO(csv_text))


def test_tencent_field_order_and_units_drop_future_and_invalid(tmp_path, monkeypatch):
    rows = [
        _tencent_row("2026-11-18", "1400.000", "1410.000", "1420.000", "1390.000", "100.000"),
        _tencent_row("2026-11-19", "1442.000", "1460.210", "1464.990", "1432.100", "42692.000"),
        _tencent_row("2026-11-20", "1", "999999", "1", "1", "1"),
        _tencent_row("2026-11-17", "10", "", "10", "10", "10"),
        _tencent_row("2026-11-16", "0", "0", "0", "0", "5"),
        _tencent_row("2026-11-15", "10", "10", "10", "10", ""),
        _tencent_row("2026-11-14", "8", "9", "10", "8", "0"),
    ]
    _use_tencent(monkeypatch, tmp_path, rows)
    result = a_stock.get_stock_data("600519", "2026-11-14", "2026-11-19")

    assert "OHLCV request cannot be fulfilled" not in result
    assert "source: tencent" in result
    assert "none (不复权, unadjusted)" in result
    assert "手 (1手=100股)" in result
    assert "As-of cutoff: 2026-11-19" in result
    assert "Asia/Shanghai" in result
    assert "Bars after cutoff dropped: 1" in result
    assert "Invalid bars dropped" in result
    assert "not filled with 0" in result
    assert "999999" not in result
    assert "请检查网络连接" not in result

    table = _csv_table(result)
    assert list(table["Date"]) == ["2026-11-14", "2026-11-18", "2026-11-19"]
    last = table.iloc[-1]
    assert last["Open"] == pytest.approx(1442.0)
    assert last["High"] == pytest.approx(1464.99)
    assert last["Low"] == pytest.approx(1432.1)
    assert last["Close"] == pytest.approx(1460.21)
    assert last["Volume"] == pytest.approx(42692)
    halted = table.loc[table["Date"] == "2026-11-14"].iloc[0]
    assert halted["Volume"] == pytest.approx(0)
    assert 0 not in set(table.loc[table["Date"] != "2026-11-14", "Close"])


def test_future_bar_after_shanghai_midnight_is_dropped():
    # 2026-11-19 16:00 UTC is 2026-11-20 00:00 in Shanghai, so it is after
    # a 2026-11-19 cutoff even though the UTC calendar date is still the 19th.
    payload = {
        "code": 0,
        "data": {
            "sh600519": {
                "day": [
                    _tencent_row("2026-11-19", "10", "11", "12", "9", "3"),
                    _tencent_row("2026-11-19T16:00:00Z", "10", "11", "12", "9", "4"),
                    _tencent_row(str(int(datetime(2026, 11, 19, 16, tzinfo=UTC).timestamp())), "10", "11", "12", "9", "5"),
                ]
            }
        },
    }
    frame, meta = parse_tencent_daily_payload(
        payload,
        symbol="sh600519",
        adjust="none",
        end_date="2026-11-19",
        param="sh600519,day,,2026-11-19,800,",
    )
    assert list(frame["Date"].dt.strftime("%Y-%m-%d")) == ["2026-11-19"]
    assert frame.iloc[0]["Volume"] == pytest.approx(3)
    assert meta["bars_after_cutoff_dropped"] == 2
    assert shanghai_calendar_date("2026-11-19T16:00:00Z") == pd.Timestamp("2026-11-20")


def test_qfq_keeps_nonpositive_price_and_selects_qfq_series(tmp_path, monkeypatch):
    rows = [
        _tencent_row("2026-11-18", "-1.5", "-1.2", "-1.0", "-2.0", "10"),
        _tencent_row("2026-11-19", "10", "11", "12", "9", "20"),
    ]
    calls = _use_tencent(monkeypatch, tmp_path, rows, adjust="qfq")
    result = a_stock.get_stock_data("600519", "2026-11-18", "2026-11-19")
    assert calls and calls[0].endswith(",qfq")
    assert "qfqday" in result
    assert "qfq (前复权" in result
    assert "non-positive price" in result
    table = _csv_table(result)
    assert table.iloc[0]["Close"] == pytest.approx(-1.2)


def test_qfq_does_not_silently_use_unadjusted_series():
    payload = {"code": 0, "data": {"sh600519": {"day": [
        _tencent_row("2026-11-19", "10", "11", "12", "9", "3")
    ]}}}
    with pytest.raises(a_stock.OhlcvFetchError) as caught:
        parse_tencent_daily_payload(
            payload,
            symbol="sh600519",
            adjust="qfq",
            end_date="2026-11-19",
            param="x",
        )
    assert "qfqday" in str(caught.value)
    assert "11" not in str(caught.value).split("keys=")[-1]


def test_http_500_empty_body_fails_clearly_without_retry(tmp_path, monkeypatch):
    calls = _use_tencent(monkeypatch, tmp_path, [], status=500, body=b"")
    result = a_stock.get_stock_data("600519", "2026-11-01", "2026-11-19")
    assert len(calls) == 1
    assert "OHLCV request cannot be fulfilled" in result
    assert "status=500" in result
    assert "body_length=0" in result
    assert "请检查网络连接" not in result
    assert "Date,Open,High,Low,Close,Volume" not in result


def test_http_200_invalid_json_reports_decode_error(tmp_path, monkeypatch):
    _use_tencent(monkeypatch, tmp_path, [], status=200, body=b"")
    empty = a_stock.get_stock_data("600519", "2026-11-01", "2026-11-19")
    assert "body_length=0" in empty
    assert "not parsed as prices" in empty

    _use_tencent(monkeypatch, tmp_path, [], status=200, body=b"<html>not json")
    broken = a_stock.get_stock_data("600519", "2026-11-01", "2026-11-19")
    assert "JSONDecodeError" in broken
    assert "body_length=" in broken


def test_captcha_refusal_is_not_retried(tmp_path, monkeypatch):
    calls = _use_tencent(
        monkeypatch,
        tmp_path,
        [],
        status=403,
        body=b'{"msg":"hit risk","isCaptchaEnabled":true}',
    )
    result = a_stock.get_stock_data("600519", "2026-11-01", "2026-11-19")
    assert len(calls) == 1
    assert "status=403" in result
    assert "Not retried" in result
    assert "hit risk" in result or "captcha" in result


def test_sina_http_error_names_status_instead_of_a_generic_network_blame(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _Resp(500, b"")

    monkeypatch.setattr(a_stock, "_http_get_public", fake_get)
    monkeypatch.setenv("ASTOCK_OHLCV_SOURCE", "sina")
    monkeypatch.setenv("ASTOCK_OHLCV_ADJUST", "none")
    monkeypatch.setattr(
        "tradingagents.dataflows.config.get_config",
        lambda: {"data_cache_dir": str(tmp_path)},
    )
    result = a_stock.get_stock_data("600519", "2026-11-01", "2026-11-19")
    assert len(calls) == 1
    assert "getKLineData" in calls[0]
    assert "source=sina" in result
    assert "status=500" in result
    assert "body_length=0" in result
    assert "请检查网络连接" not in result


def test_historical_cache_does_not_refetch_or_invent_bars(tmp_path, monkeypatch):
    rows = [_tencent_row("2026-11-19", "1442", "1460.21", "1464.99", "1432.1", "42692")]
    calls = _use_tencent(monkeypatch, tmp_path, rows)
    first = a_stock.get_stock_data("600519", "2026-11-19", "2026-11-19")
    second = a_stock.get_stock_data("600519", "2026-11-19", "2026-11-19")
    assert len(calls) == 1
    assert "# Cache:" not in first
    assert "# Cache:" in second
    assert _csv_table(second).iloc[0]["Close"] == pytest.approx(1460.21)


def test_default_source_is_tencent_none(monkeypatch):
    monkeypatch.delenv("ASTOCK_OHLCV_SOURCE", raising=False)
    monkeypatch.delenv("ASTOCK_OHLCV_ADJUST", raising=False)
    monkeypatch.setattr("tradingagents.dataflows.config.get_config", lambda: {})
    assert a_stock._ohlcv_settings() == ("tencent", "none")


def test_indicators_receive_cutoff_bars_and_stockstats_values(tmp_path, monkeypatch):
    end = datetime(2026, 11, 19)
    rows = []
    for offset in range(59, -1, -1):
        day = (end - timedelta(days=offset)).strftime("%Y-%m-%d")
        price = 100 + (59 - offset)
        rows.append(_tencent_row(day, price, price, price + 1, price - 1, 1000 + offset))
    rows.append(_tencent_row("2026-11-20", "1", "999999", "1", "1", "1"))
    rows.append(_tencent_row("2026-11-21", "", "", "", "", ""))
    from stockstats import wrap as real_wrap

    _use_tencent(monkeypatch, tmp_path, rows)

    captured = []

    def fake_wrap(data):
        captured.append(data.copy())
        out = data.copy()
        out["rsi"] = out["Close"].astype(float)
        return out

    monkeypatch.setattr("stockstats.wrap", fake_wrap)
    preview = a_stock.get_indicators("600519", "rsi", "2026-11-19", 0)
    assert captured, preview
    handed = captured[0]
    assert handed["Date"].max() == pd.Timestamp("2026-11-19")
    assert pd.Timestamp("2026-11-20") not in set(handed["Date"])
    assert 999999 not in set(handed["Close"])
    assert (handed["Close"] == 0).sum() == 0
    assert handed["Volume"].min() > 0
    assert "手 (1手=100股)" in preview
    assert "As-of cutoff: 2026-11-19" in preview
    assert "stockstats" in preview
    assert "Bars after cutoff dropped: 1" in preview

    # The same cleaned bars, run through stockstats itself, must match the tool.
    # That keeps the indicator algorithm on stockstats.
    monkeypatch.setattr("stockstats.wrap", real_wrap)
    expected = real_wrap(handed.copy())
    expected["close_50_sma"]
    expected_dates = pd.to_datetime(expected["Date"])
    last = expected.loc[expected_dates == pd.Timestamp("2026-11-19"), "close_50_sma"]
    assert not last.empty
    expected_text = str(round(float(last.iloc[0]), 4))
    output = a_stock.get_indicators("600519", "close_50_sma", "2026-11-19", 0)
    assert f"2026-11-19: {expected_text}" in output
    assert "999999" not in output
    assert "Indicators were not computed" not in output


def test_indicator_http_failure_does_not_compute(tmp_path, monkeypatch):
    _use_tencent(monkeypatch, tmp_path, [], status=500, body=b"")
    result = a_stock.get_indicators("600519", "rsi", "2026-11-19", 3)
    assert "status=500" in result
    assert "body_length=0" in result
    assert "Indicators were not computed" in result
    assert "No prices were filled with zeros" in result


def test_news_timestamp_unix_and_aware_share_shanghai_clock():
    wall = datetime(2026, 11, 19, 23, 0, tzinfo=SH)
    unix_s = int(wall.timestamp())
    unix_ms = unix_s * 1000
    aware = wall.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    zulu = wall.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    parsed = [
        parse_news_timestamp(unix_s),
        parse_news_timestamp(str(unix_s)),
        parse_news_timestamp(unix_ms),
        parse_news_timestamp(aware),
        parse_news_timestamp(zulu),
        parse_news_timestamp("2026-11-19 23:00:00"),
        parse_news_timestamp("2026-11-19 23:00"),
    ]
    assert all(item is not None for item in parsed)
    assert {item.instant for item in parsed} == {wall}
    assert all(item.instant.utcoffset() == timedelta(hours=8) for item in parsed)


def test_news_day_boundary_unknown_times_and_ignored_title_dates():
    items = [
        {"title": "kept at end of day", "time": "2026-11-19 23:59:59", "source": "东方财富"},
        {"title": "kept exact as-of microsecond", "time": "2026-11-19T23:59:59.999999+08:00", "source": "东方财富"},
        {"title": "next midnight shanghai", "time": "2026-11-20 00:00:00", "source": "东方财富"},
        {"title": "utc evening is next shanghai day", "time": "2026-11-19T16:00:00Z", "source": "CLS Wire"},
        {
            "title": "unix next midnight",
            "time": int(datetime(2026, 11, 20, 0, 0, tzinfo=SH).timestamp()),
            "source": "CLS Wire",
        },
        {"title": "before window", "time": "2026-11-17 23:59:59", "source": "新浪财经"},
        {"title": "date only on as-of day", "time": "2026-11-19", "source": "东方财富"},
        {"title": "date only next day", "time": "2026-11-20", "source": "东方财富"},
        {"title": "headline says 2099-01-01", "time": "2026-11-18 09:00:00", "source": "东方财富", "url": "https://x/20261120"},
        {"title": "unknown word today", "time": "today", "source": "东方财富"},
        {"title": "blank", "time": "", "source": "东方财富"},
        {"title": "bool must not be unix 1", "time": True, "source": "东方财富"},
        {"title": "realSort is not a timestamp", "time": 1795190232034983, "source": "Eastmoney Global"},
        {
            "title": "headline says 2099-01-01",
            "time": "2026-11-18 09:00:00",
            "source": "东方财富",
            "url": "https://x/20261120",
        },
    ]
    filtered = filter_news_items(items, "2026-11-18", "2026-11-19")
    titles = [row["title"] for row in filtered.kept]
    assert "next midnight shanghai" not in titles
    assert "utc evening is next shanghai day" not in titles
    assert "unix next midnight" not in titles
    assert "before window" not in titles
    assert "date only next day" not in titles
    assert "unknown word today" not in titles
    assert "bool must not be unix 1" not in titles
    assert "realSort is not a timestamp" not in titles
    assert titles.count("headline says 2099-01-01") == 1
    assert "kept at end of day" in titles
    assert "kept exact as-of microsecond" in titles
    assert "date only on as-of day" in titles
    assert filtered.dropped_after_asof == 4
    assert filtered.dropped_before_window == 1
    assert filtered.dropped_unparseable == 4
    assert filtered.dropped_duplicate == 1
    date_only = next(row for row in filtered.kept if row["title"] == "date only on as-of day")
    assert date_only["precision"] == "date"
    assert date_only["published_at"] == datetime(2026, 11, 19, 0, 0, tzinfo=SH)


def test_get_news_reports_drops_and_tries_backup_when_primary_is_all_future(monkeypatch):
    calls = []

    def eastmoney(_code):
        calls.append("eastmoney")
        return [
            {"title": "still in the future", "time": "2026-11-20 00:00:01", "source": "东方财富", "content": "later"},
            {"title": "no clock", "time": None, "source": "东方财富", "content": "missing"},
        ]

    def sina(_code):
        calls.append("sina")
        return [
            {
                "title": "usable wire",
                "time": "2026-11-19 15:04:00",
                "source": "新浪财经",
                "content": "body",
                "url": "https://example.test/a",
            }
        ]

    monkeypatch.setattr(a_stock, "_fetch_news_eastmoney", eastmoney)
    monkeypatch.setattr(a_stock, "_fetch_news_sina", sina)
    result = a_stock.get_news("600519", "2026-11-18", "2026-11-19")
    assert calls == ["eastmoney", "sina"]
    assert "As-of exclusion filter: enforceable." in result
    assert "NOT enforceable" not in result
    assert "after_asof=1" in result
    assert "unparseable=1" in result
    assert "2026-11-19 15:04:00 Asia/Shanghai | 新浪财经 | usable wire" in result
    assert "- source: 新浪财经" in result
    assert "still in the future" not in result
    assert "no clock" not in result
    assert "complete historical forecast" in result
    assert "not a point-in-time archive" in result


def test_get_news_stops_after_primary_keeps_an_item(monkeypatch):
    calls = []

    def eastmoney(_code):
        calls.append("eastmoney")
        return [{"title": "in window", "time": "2026-11-19 10:00:00", "source": "东方财富"}]

    def sina(_code):
        calls.append("sina")
        return [{"title": "should not be fetched", "time": "2026-11-19 11:00:00", "source": "新浪财经"}]

    monkeypatch.setattr(a_stock, "_fetch_news_eastmoney", eastmoney)
    monkeypatch.setattr(a_stock, "_fetch_news_sina", sina)
    result = a_stock.get_news("600519", "2026-11-19", "2026-11-19")
    assert calls == ["eastmoney"]
    assert "should not be fetched" not in result
    assert "in window" in result


def test_get_news_all_sources_failed_is_not_enforceable(monkeypatch):
    def boom(_code):
        raise RuntimeError("HTTP 500 body_length=0")

    monkeypatch.setattr(a_stock, "_fetch_news_eastmoney", boom)
    monkeypatch.setattr(a_stock, "_fetch_news_sina", boom)
    result = a_stock.get_news("600519", "2026-11-19", "2026-11-19")
    assert "As-of exclusion filter: NOT enforceable." in result
    assert "HTTP 500 body_length=0" in result
    assert "Do not use this result as a historical news set" in result
    assert "### " not in result


def test_global_news_filters_cls_unix_and_eastmoney_showtime(monkeypatch):
    in_window = int(datetime(2026, 11, 19, 23, 0, tzinfo=SH).timestamp())
    next_midnight = int(datetime(2026, 11, 20, 0, 0, tzinfo=SH).timestamp())

    def fake_get(url, params=None, headers=None, timeout=None):
        assert "telegraphList" in url
        return _Resp(
            200,
            json.dumps(
                {
                    "data": {
                        "roll_data": [
                            {"title": "cls kept", "content": "a", "ctime": in_window, "brief": ""},
                            {"title": "cls future", "content": "b", "ctime": next_midnight},
                            {"title": "cls unknown", "content": "c", "ctime": "not-a-time"},
                            {"title": "cls kept", "content": "a", "ctime": in_window},
                        ]
                    }
                }
            ).encode(),
        )

    def fake_em(url, params=None, headers=None, timeout=None):
        assert "getFastNewsList" in url
        return _Resp(
            200,
            json.dumps(
                {
                    "data": {
                        "fastNewsList": [
                            {
                                "title": "em kept",
                                "summary": "s",
                                "showTime": "2026-11-18 00:00:00",
                            },
                            {
                                "title": "em day boundary",
                                "summary": "s",
                                "showTime": "2026-11-19T16:00:00Z",
                            },
                            {"title": "em blank", "summary": "s", "showTime": ""},
                        ]
                    }
                }
            ).encode(),
        )

    monkeypatch.setattr(a_stock._requests, "get", fake_get)
    monkeypatch.setattr(a_stock, "_em_get", fake_em)
    result = a_stock.get_global_news("2026-11-19", look_back_days=1, limit=10)
    assert "As-of exclusion filter: enforceable." in result
    assert "2026-11-18 00:00:00 Asia/Shanghai | Eastmoney Global | em kept" in result
    assert "2026-11-19 23:00:00 Asia/Shanghai | CLS Wire | cls kept" in result
    assert "cls future" not in result
    assert "em day boundary" not in result
    assert "em blank" not in result
    assert "after_asof=2" in result
    assert "unparseable=2" in result
    assert "duplicate=1" in result
    assert "before_window=0" in result
    assert result.count("### cls kept") == 1
    assert "CLS Wire" in result and "Eastmoney Global" in result


def test_global_news_limit_keeps_newest_and_reports_omission(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        return _Resp(200, json.dumps({"data": {"roll_data": []}}).encode())

    def fake_em(url, params=None, headers=None, timeout=None):
        return _Resp(
            200,
            json.dumps(
                {
                    "data": {
                        "fastNewsList": [
                            {"title": "older", "summary": "", "showTime": "2026-11-18 01:00:00"},
                            {"title": "newer", "summary": "", "showTime": "2026-11-19 01:00:00"},
                        ]
                    }
                }
            ).encode(),
        )

    monkeypatch.setattr(a_stock._requests, "get", fake_get)
    monkeypatch.setattr(a_stock, "_em_get", fake_em)
    result = a_stock.get_global_news("2026-11-19", look_back_days=7, limit=1)
    assert "Kept: 2 shown: 1 omitted_by_limit=1" in result
    assert "newer" in result
    assert "older" not in result.split("###", 1)[-1] or "### older" not in result
    assert "### newer" in result
    assert "### older" not in result


def test_global_news_failed_branches_are_not_a_forecast(monkeypatch):
    def fail_get(url, params=None, headers=None, timeout=None):
        return _Resp(404, b"")

    def fail_em(url, params=None, headers=None, timeout=None):
        return _Resp(403, b'{"msg":"hit risk","isCaptchaEnabled":true}')

    monkeypatch.setattr(a_stock._requests, "get", fail_get)
    monkeypatch.setattr(a_stock, "_em_get", fail_em)
    result = a_stock.get_global_news("2026-11-19", 7, 5)
    assert "NOT enforceable" in result
    assert "CLS Wire" in result and "body_length=0" in result
    assert "Eastmoney Global" in result and "Not retried" in result
    assert "Do not use this result as a historical news set" in result
