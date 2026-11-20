"""Process-visible run status.

Debug streaming pretty-prints only the last message of a graph chunk, and a
callback ``on_tool_start`` is not a reliable signal that a ToolNode batch has
entered a tool. This module writes start, end, and failure lines that are
flushed immediately.

If the process is signaled or exits while the status is still ``running``,
the status file is rewritten to ``failed`` with a reason. ``SIGKILL`` cannot
be caught.
"""

from __future__ import annotations

import atexit
import faulthandler
import os
import signal
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_SHANGHAI = timezone(timedelta(hours=8))
_lock = threading.Lock()
_status_path: Path | None = None
_event_path: Path | None = None
_fault_file = None
_installed = False
_previous: dict[int, Any] = {}
_state: dict[str, str] = {"status": "idle", "phase": "", "reason": ""}


def configure(status_path: Path, event_path: Path | None = None) -> None:
    """Point status files somewhere other than ``./logs`` (tests)."""
    global _status_path, _event_path
    _status_path = Path(status_path)
    _event_path = Path(event_path) if event_path is not None else _status_path.with_name(
        "runtime_status.log"
    )


def _paths() -> tuple[Path, Path]:
    global _status_path, _event_path
    if _status_path is None or _event_path is None:
        directory = Path("logs")
        _status_path = directory / "RUN_STATUS.md"
        _event_path = directory / "runtime_status.log"
    return _status_path, _event_path


def _now() -> str:
    return datetime.now(_SHANGHAI).isoformat(timespec="seconds")


def _compose(status: str, phase: str, reason: str) -> str:
    lines = [
        f"status: {status}",
        f"phase: {phase or 'unknown'}",
        f"updated_at: {_now()}",
    ]
    if reason:
        lines.append(f"reason: {reason}")
    lines.append("")
    return "\n".join(lines)


def _write(status: str, *, phase: str, reason: str = "") -> None:
    """Write the status file. Takes the lock with a timeout so a signal cannot deadlock."""
    got = _lock.acquire(timeout=0.2)
    try:
        _state["status"] = status
        _state["phase"] = phase
        _state["reason"] = reason
        path, _event = _paths()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(_compose(status, phase, reason), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        print(f"runtime status write failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        if got:
            _lock.release()


def _append_event(line: str) -> None:
    text = f"{_now()} {line}\n"
    print(text, file=sys.stderr, end="", flush=True)
    try:
        _status, event_path = _paths()
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
    except Exception as exc:
        print(f"runtime status event failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def _ensure_fault_file() -> None:
    global _fault_file
    if _fault_file is not None:
        return
    try:
        _status, event_path = _paths()
        event_path.parent.mkdir(parents=True, exist_ok=True)
        _fault_file = event_path.open("a", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception as exc:
        print(f"faulthandler enable failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def mark_running(phase: str) -> None:
    activate()
    _write("running", phase=phase)
    _append_event(f"STATUS running phase={phase}")


def mark_finished(status: str, *, phase: str, reason: str = "") -> None:
    """``status`` is ``completed`` or ``failed``."""
    activate()
    _write(status, phase=phase, reason=reason)
    _append_event(f"STATUS {status} phase={phase} reason={reason}")


def note_tool_start(stage: str, tool_name: str, args: Any = None) -> None:
    activate()
    phase = f"tool {stage}/{tool_name or 'unknown'}"
    _write("running", phase=phase)
    _append_event(f"TOOL START stage={stage} tool={tool_name or 'unknown'} args={_safe_args(args)}")


def note_tool_end(
    stage: str,
    tool_name: str,
    *,
    ok: bool,
    elapsed_s: float,
    detail: str = "",
) -> None:
    # A finished tool does not mean the process is finished. Leave status running
    # so a later signal or interpreter exit still records a failure reason.
    # Do not overwrite a terminal failed/completed status written by a signal.
    phase = f"tool {stage}/{tool_name or 'unknown'} {'ok' if ok else 'failed'}"
    if _state.get("status") not in {"failed", "completed"}:
        _write("running", phase=phase)
    label = "TOOL END" if ok else "TOOL FAIL"
    _append_event(
        f"{label} stage={stage} tool={tool_name or 'unknown'} "
        f"elapsed_s={elapsed_s:.3f} detail={detail}"
    )


def note_signal(signum: int) -> None:
    try:
        name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        name = str(signum)
    phase = _state.get("phase") or "unknown"
    reason = f"received {name} while phase was {phase}"
    _write("failed", phase=phase, reason=reason)
    _append_event(f"SIGNAL {reason}")


def _on_exit() -> None:
    if _state.get("status") != "running":
        return
    phase = _state.get("phase") or "unknown"
    reason = (
        f"process exited while status was still running ({phase}). "
        "No clean completion was recorded."
    )
    _write("failed", phase=phase, reason=reason)
    _append_event(f"ATEXIT {reason}")


def _handle_signal(signum: int, frame) -> None:
    note_signal(signum)
    previous = _previous.get(signum, signal.SIG_DFL)
    if previous is signal.SIG_IGN:
        return
    if callable(previous) and previous is not _handle_signal:
        previous(signum, frame)
        return
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def activate() -> None:
    """Install atexit, SIGINT, and SIGTERM handlers once."""
    global _installed
    _paths()
    _ensure_fault_file()
    if _installed:
        return
    atexit.register(_on_exit)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            _previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            # Signal registration is only available on the main thread.
            pass
    _installed = True


def _safe_args(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    redacted = []
    for key, value in args.items():
        lowered = str(key).lower()
        if any(word in lowered for word in ("key", "token", "secret", "password", "credential")):
            redacted.append(f"{key}=***")
        else:
            redacted.append(f"{key}={value}")
    return ",".join(redacted)
