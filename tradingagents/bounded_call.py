"""Run a blocking call with a wall-clock cap.

The worker is a daemon thread. When the cap is hit the caller continues;
the worker is not killed, but it cannot keep the process alive on shutdown.
"""

from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")


class DeadlineExceeded(TimeoutError):
    """The wall-clock cap elapsed before the call returned."""


def call_with_timeout(fn: Callable[[], T], timeout: float) -> T:
    """Return ``fn()`` or raise ``DeadlineExceeded`` when ``timeout`` elapses.

    A ``DeadlineExceeded`` raised by ``fn`` itself is propagated unchanged.
    Only the wait here produces a timeout that the caller did not raise.
    """
    if timeout <= 0:
        raise DeadlineExceeded(f"timed out after {timeout:.1f}s")

    box: dict[str, object] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # transferred to the caller thread
            box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=run, name="ta-bounded-call", daemon=True)
    worker.start()
    if not done.wait(timeout):
        raise DeadlineExceeded(f"timed out after {timeout:.1f}s")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]
