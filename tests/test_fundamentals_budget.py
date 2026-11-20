"""Fundamentals must not wait on a mootdx server hunt.

2026-11-21: 14 servers accepted TCP 7709 and none returned data.
``_get_mootdx_client()`` took 157.730s, and LangGraph's ToolNode held the
other five fundamental tools until that hunt finished. These tests lock the
wall-clock caps and the Eastmoney as-of filter.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from datetime import date
from typing import Annotated
from unittest.mock import MagicMock

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import pytest
import requests

from tradingagents.bounded_call import DeadlineExceeded
from tradingagents.dataflows import a_stock
from tradingagents.dataflows import missing_data
from tradingagents import runtime_status


AS_OF = "2026-11-19"
KEPT_REVENUE = "133283652508.95"
DROPPED_REVENUE = "999888777.66"


class _ToolState(TypedDict):
    messages: Annotated[list, add_messages]


def _row(**overrides) -> dict:
    row = {
        "SECURITY_CODE": "600519",
        "SECURITY_NAME_ABBR": "贵州茅台",
        "REPORT_DATE": "2026-09-30 00:00:00",
        "REPORT_TYPE": "三季报",
        "NOTICE_DATE": "2026-10-31 00:00:00",
        "UPDATE_DATE": "2026-10-31 00:00:00",
        "EPSJB": "51.53",
        "BPS": "216.949629787912",
        "TOTALOPERATEREVE": KEPT_REVENUE,
        "PARENTNETPROFIT": "64467425981.95",
        "ROEJQ": "24.47",
        "ZCFZL": "11.9015685357",
    }
    row.update(overrides)
    return row


def _future_row() -> dict:
    return _row(
        REPORT_DATE="2026-12-31 00:00:00",
        REPORT_TYPE="年报",
        NOTICE_DATE="2026-11-20 00:00:00",
        UPDATE_DATE="2026-11-20 00:00:00",
        TOTALOPERATEREVE=DROPPED_REVENUE,
        PARENTNETPROFIT="1.00",
    )


class _HTTP:
    def __init__(self, status: int, body: bytes):
        self.status_code = status
        self.content = body
        self.text = body.decode("utf-8", errors="replace")

    def json(self):
        if not self.content.strip():
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return json.loads(self.content)


def _mainfina_response(*rows: dict) -> _HTTP:
    payload = {
        "success": True,
        "result": {"pages": 1, "data": list(rows)},
    }
    return _HTTP(200, json.dumps(payload).encode("utf-8"))


@pytest.fixture
def isolated_status(tmp_path):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    previous_state = dict(runtime_status._state)
    runtime_status.configure(tmp_path / "RUN_STATUS.md", tmp_path / "runtime_status.log")
    runtime_status._state.update({"status": "idle", "phase": "", "reason": ""})
    yield tmp_path / "RUN_STATUS.md"
    for sig, handler in previous.items():
        signal.signal(sig, handler)
    runtime_status._state.update(previous_state)
    runtime_status._state["status"] = "completed"


def test_budget_constants_are_inside_the_agreed_range():
    assert 8 <= a_stock._MOOTDX_CLIENT_BUDGET_S <= 15
    assert 5 <= a_stock._MOOTDX_FINANCE_FALLBACK_S <= 8
    assert 1 <= a_stock._MOOTDX_FACTORY_TIMEOUT_S < 15
    assert 20 <= missing_data.TOOL_HARD_TIMEOUT_S <= 30


def test_notice_on_asof_day_is_kept_and_later_notice_is_dropped():
    as_of = date(2026, 11, 19)
    assert a_stock._mainfina_in_window(_row(), as_of)
    assert a_stock._mainfina_in_window(
        _row(NOTICE_DATE="2026-11-19 00:00:00", UPDATE_DATE="2026-11-19 00:00:00"),
        as_of,
    )
    assert not a_stock._mainfina_in_window(_future_row(), as_of)
    assert not a_stock._mainfina_in_window(
        _row(NOTICE_DATE=None, UPDATE_DATE=None),
        as_of,
    )
    # A report period after the as-of date is not public yet, even with an early notice.
    assert not a_stock._mainfina_in_window(
        _row(REPORT_DATE="2026-12-31 00:00:00", NOTICE_DATE="2026-10-31 00:00:00"),
        as_of,
    )


def test_later_update_is_flagged_but_notice_still_gates():
    as_of = date(2026, 11, 19)
    row = _row(UPDATE_DATE="2026-11-20 00:00:00")
    assert a_stock._mainfina_in_window(row, as_of)
    text = a_stock._format_mainfina_row(row, as_of, dropped=0)
    assert "UPDATE_DATE is after" in text
    assert KEPT_REVENUE in text


def test_get_fundamentals_returns_eastmoney_without_waiting_for_mootdx(monkeypatch):
    mootdx_calls = []

    def slow_mootdx(*_args, **_kwargs):
        mootdx_calls.append(time.monotonic())
        time.sleep(30)
        raise AssertionError("mootdx hunt should not run when Eastmoney has an in-window row")

    def fake_em(url, params=None, timeout=15, **_kwargs):
        if "securities/api/data" in str(url) or (
            isinstance(params, dict) and params.get("reportName") == a_stock._MAINFINA_REPORT
        ):
            return _mainfina_response(_future_row(), _row())
        return _HTTP(502, b"<html>502</html>")

    monkeypatch.setattr(a_stock, "_tencent_quote", lambda codes, timeout=10: {
        "600519": {
            "name": "贵州茅台",
            "price": 1485.75,
            "pe_ttm": 22.61,
            "pe_static": 22.56,
            "pb": 6.85,
            "mcap_yi": 18573.09,
            "float_mcap_yi": 18573.09,
            "turnover_pct": 0.54,
            "change_pct": 1.75,
            "limit_up": 1606.23,
            "limit_down": 1314.19,
        }
    })
    monkeypatch.setattr(a_stock, "_em_get", fake_em)
    monkeypatch.setattr(a_stock, "_mootdx_call", slow_mootdx)
    monkeypatch.setattr(a_stock, "_get_mootdx_client", slow_mootdx)
    monkeypatch.setattr(
        a_stock,
        "_ths_eps_forecast",
        lambda code, timeout=6: (_ for _ in ()).throw(
            RuntimeError("[数据缺失: 一致预期EPS] HTTP status=500 body_length=0")
        ),
    )

    started = time.monotonic()
    out = a_stock.get_fundamentals("600519", AS_OF)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"get_fundamentals waited {elapsed:.2f}s"
    assert mootdx_calls == []
    assert KEPT_REVENUE in out
    assert "2026-09-30" in out
    assert "2026-10-31" in out
    assert DROPPED_REVENUE not in out
    assert "2026-12-31" not in out
    assert "RPT_F10_FINANCE_MAINFINADATA" in out
    assert "未来函数警告" in out
    assert "当前快照" in out
    assert "不是" in out and "历史" in out
    assert "[数据缺失: 一致预期EPS]" in out
    assert "[数据缺失: 东财个股信息]" in out
    assert "502" in out


def test_mootdx_fallback_cannot_block_tencent_when_eastmoney_is_empty(monkeypatch):
    release = threading.Event()
    calls = []

    def slow_mootdx(*_args, **_kwargs):
        calls.append("mootdx")
        release.wait(30)
        return None

    def fake_em(url, params=None, timeout=15, **_kwargs):
        if isinstance(params, dict) and params.get("reportName") == a_stock._MAINFINA_REPORT:
            return _mainfina_response(_future_row())
        return _HTTP(200, b"")

    monkeypatch.setattr(a_stock, "_MOOTDX_FINANCE_FALLBACK_S", 0.7)
    monkeypatch.setattr(a_stock, "_tencent_quote", lambda codes, timeout=10: {
        "600519": {
            "name": "贵州茅台",
            "price": 1460.21,
            "pe_ttm": 20.0,
            "pe_static": 20.0,
            "pb": 6.0,
            "mcap_yi": 1.0,
            "float_mcap_yi": 1.0,
            "turnover_pct": 0.1,
            "change_pct": 0.1,
            "limit_up": 1.0,
            "limit_down": 1.0,
        }
    })
    monkeypatch.setattr(a_stock, "_em_get", fake_em)
    monkeypatch.setattr(a_stock, "_mootdx_call", slow_mootdx)
    monkeypatch.setattr(
        a_stock,
        "_ths_eps_forecast",
        lambda code, timeout=6: (_ for _ in ()).throw(RuntimeError("ths down")),
    )

    started = time.monotonic()
    try:
        out = a_stock.get_fundamentals("600519", AS_OF)
    finally:
        release.set()
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, f"mootdx fallback blocked for {elapsed:.2f}s"
    assert calls == ["mootdx"]
    assert "1460.21" in out
    assert "[数据缺失: mootdx财务快照]" in out
    assert DROPPED_REVENUE not in out
    assert "timed out" in out.lower() or "DeadlineExceeded" in out


def test_get_mootdx_client_stops_at_wall_clock_budget(monkeypatch, tmp_path):
    import mootdx.config
    import mootdx.quotes

    # setup() reads CONF, captured at import. A missing file makes it call
    # bestip() and speed-test hosts before our budget loop. Point CONF at an
    # empty path and prove that scan does not run.
    config_path = tmp_path / "mootdx-config.json"
    monkeypatch.setattr(mootdx.config, "CONF", str(config_path))

    release = threading.Event()
    servers = [(f"10.0.0.{i}", 7709) for i in range(1, 15)]
    state = {"factory": [], "timeouts": []}
    monkeypatch.setattr(a_stock, "_TDX_SERVERS", servers)
    monkeypatch.setattr(a_stock, "_candidate_tdx_servers", lambda: list(servers))
    monkeypatch.setattr(a_stock, "_mootdx_client", None)
    monkeypatch.setattr(a_stock, "_mootdx_unavailable_until", 0.0)
    monkeypatch.setattr(a_stock, "_probe_tdx", lambda ip, port, timeout=2.0: True)

    class SlowQuotes:
        @staticmethod
        def factory(market="std", server=None, **kwargs):
            state["factory"].append(server[0] if server else "bare")
            state["timeouts"].append(kwargs.get("timeout"))
            release.wait(30)
            raise TimeoutError("slow server")

    monkeypatch.setattr(mootdx.quotes, "Quotes", SlowQuotes)

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="预算"):
            a_stock._get_mootdx_client(budget_s=1.2)
    finally:
        release.set()
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, f"_get_mootdx_client ran for {elapsed:.2f}s"
    assert len(state["factory"]) == 1
    assert state["timeouts"][0] == 1
    assert all(isinstance(item, int) and 1 <= item < 15 for item in state["timeouts"])

    before = len(state["factory"])
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="不再重试"):
        a_stock._get_mootdx_client()
    assert time.monotonic() - started < 0.5
    assert len(state["factory"]) == before


def test_factory_timeout_is_explicit_on_the_fast_failure_path(monkeypatch):
    import mootdx.quotes

    servers = [("1.1.1.1", 7709)]
    seen = []
    monkeypatch.setattr(a_stock, "_TDX_SERVERS", servers)
    monkeypatch.setattr(a_stock, "_candidate_tdx_servers", lambda: list(servers))
    monkeypatch.setattr(a_stock, "_mootdx_client", None)
    monkeypatch.setattr(a_stock, "_mootdx_unavailable_until", 0.0)
    monkeypatch.setattr(a_stock, "_probe_tdx", lambda ip, port, timeout=2.0: True)

    class FakeQuotes:
        @staticmethod
        def factory(market="std", server=None, **kwargs):
            seen.append(kwargs.get("timeout"))
            raise ConnectionResetError("reset")

    monkeypatch.setattr(mootdx.quotes, "Quotes", FakeQuotes)
    with pytest.raises(RuntimeError, match="协议握手/取数被拒"):
        a_stock._get_mootdx_client()
    assert seen
    assert all(isinstance(item, int) and 1 <= item <= a_stock._MOOTDX_FACTORY_TIMEOUT_S for item in seen)


def test_slow_tool_does_not_block_the_rest_of_the_batch(monkeypatch, isolated_status):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    monkeypatch.setattr(missing_data, "TOOL_HARD_TIMEOUT_S", 0.6)
    release = threading.Event()

    @tool
    def fast_tool(ticker: str) -> str:
        """Return immediately."""
        return f"FAST_OK {ticker}"

    @tool
    def slow_tool(ticker: str) -> str:
        """Block until released."""
        release.wait(30)
        return f"SLOW_OK {ticker}"

    node = ToolNode(
        [fast_tool, slow_tool],
        wrap_tool_call=missing_data.make_tool_call_recorder("fundamentals"),
    )
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "fast_tool", "args": {"ticker": "600519"}, "id": "call-fast", "type": "tool_call"},
            {"name": "slow_tool", "args": {"ticker": "600519"}, "id": "call-slow", "type": "tool_call"},
        ],
    )

    def supply(_state):
        return {"messages": [message]}

    graph = StateGraph(_ToolState)
    graph.add_node("supply", supply)
    graph.add_node("tools", node)
    graph.add_edge(START, "supply")
    graph.add_edge("supply", "tools")
    graph.add_edge("tools", END)

    started = time.monotonic()
    try:
        result = graph.compile().invoke({"messages": []})
    finally:
        release.set()
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, f"tool batch waited {elapsed:.2f}s for the slow tool"
    messages = result["messages"]
    by_name = {getattr(item, "name", ""): str(getattr(item, "content", "")) for item in messages}
    assert "FAST_OK 600519" in by_name["fast_tool"]
    assert "timed out" in by_name["slow_tool"].lower()
    assert "SLOW_OK" not in by_name["slow_tool"]
    event_log = isolated_status.with_name("runtime_status.log").read_text(encoding="utf-8")
    assert "TOOL START" in event_log
    assert "fast_tool" in event_log
    assert "slow_tool" in event_log
    assert "TOOL FAIL" in event_log or "timed out" in event_log


def test_exit_and_signal_do_not_leave_status_running(isolated_status):
    runtime_status.mark_running("tool fundamentals/get_fundamentals")
    assert isolated_status.read_text(encoding="utf-8").startswith("status: running")

    runtime_status._on_exit()
    failed = isolated_status.read_text(encoding="utf-8")
    assert failed.startswith("status: failed")
    assert "process exited while status was still running" in failed
    assert "tool fundamentals/get_fundamentals" in failed

    runtime_status.mark_running("tool fundamentals/get_balance_sheet")
    called = []
    runtime_status._previous[signal.SIGTERM] = lambda signum, frame: called.append(signum)
    runtime_status._handle_signal(signal.SIGTERM, None)
    signaled = isolated_status.read_text(encoding="utf-8")
    assert signaled.startswith("status: failed")
    assert "SIGTERM" in signaled
    assert called == [signal.SIGTERM]


def test_sina_retries_one_read_timeout_and_refuses_blocked_pages(monkeypatch):
    from tests.test_sina_financial_report_shape import NEW_SHAPE

    calls = {"n": 0}

    def fake_get(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ReadTimeout("read timed out")
        response = MagicMock()
        response.status_code = 200
        response.content = json.dumps(NEW_SHAPE).encode("utf-8")
        response.json.return_value = NEW_SHAPE
        return response

    monkeypatch.setattr(a_stock._requests, "get", fake_get)
    out = a_stock.get_balance_sheet("600519", "quarterly", "2026-09-30")
    assert calls["n"] == 2
    assert "货币资金" in out
    assert "publication timestamp" in out

    blocked = MagicMock()
    blocked.status_code = 403
    blocked.content = b"denied"
    blocked.text = "denied"
    monkeypatch.setattr(a_stock._requests, "get", lambda *_a, **_k: blocked)
    denied = a_stock.get_income_statement("600519", "quarterly", "2026-09-30")
    assert "403" in denied
    assert "body_length=6" in denied
    assert "Not retried" in denied

    captcha_calls = {"n": 0}

    def captcha(*_args, **_kwargs):
        captcha_calls["n"] += 1
        response = MagicMock()
        response.status_code = 200
        response.content = b"please complete the captcha"
        response.text = "please complete the captcha"
        response.json.side_effect = AssertionError("captcha body must not be parsed")
        return response

    monkeypatch.setattr(a_stock._requests, "get", captcha)
    captcha_out = a_stock.get_cashflow("600519", "quarterly", "2026-09-30")
    assert captcha_calls["n"] == 1
    assert "captcha" in captcha_out.lower() or "Blocked" in captcha_out
    assert "Not retried" in captcha_out

    connect_calls = {"n": 0}

    def connect(*_args, **_kwargs):
        connect_calls["n"] += 1
        raise requests.ConnectTimeout("connect timed out")

    monkeypatch.setattr(a_stock._requests, "get", connect)
    connect_out = a_stock.get_balance_sheet("600519", "quarterly", "2026-09-30")
    assert connect_calls["n"] == 1
    assert "ConnectTimeout" in connect_out or "connect timed out" in connect_out


def test_industry_and_consensus_fail_fast_without_a_second_source(monkeypatch):
    calls = {"em": 0, "ths": 0}

    def fake_em(*_args, **_kwargs):
        calls["em"] += 1
        return _HTTP(502, b"<html>502 Bad Gateway</html>")

    def fake_ths(*_args, **_kwargs):
        calls["ths"] += 1
        response = MagicMock()
        response.status_code = 500
        response.content = b""
        response.text = ""
        response.json.side_effect = AssertionError("error page must not be parsed as JSON")
        return response

    monkeypatch.setattr(a_stock, "_em_get", fake_em)
    monkeypatch.setattr(a_stock._requests, "get", fake_ths)

    started = time.monotonic()
    industry = a_stock.get_industry_comparison("600519", AS_OF)
    forecast = a_stock.get_profit_forecast("600519", AS_OF)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert calls["em"] == 1
    assert calls["ths"] == 1
    assert "[数据缺失: 行业对比]" in industry
    assert "502" in industry
    assert "body_length=" in industry
    assert "[数据缺失: 一致预期EPS]" in forecast
    assert "500" in forecast
    assert "未来函数警告" in forecast


def test_deadline_exceeded_is_a_timeout():
    with pytest.raises(DeadlineExceeded):
        from tradingagents.bounded_call import call_with_timeout

        call_with_timeout(lambda: time.sleep(2), 0.2)
