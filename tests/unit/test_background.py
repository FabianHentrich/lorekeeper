"""run_in_daemon_thread: blocking jobs that must not keep the process alive at exit (roadmap 12.3)."""
import threading

import pytest

from src.api.background import run_in_daemon_thread


async def test_returns_the_result_from_a_daemon_thread():
    seen = {}

    def job(a, b):
        seen["thread"] = threading.current_thread()
        return a + b

    assert await run_in_daemon_thread(job, 2, 3) == 5
    # The point of the helper: a pool thread is joined at exit and kept a stopped dev worker alive
    assert seen["thread"].daemon
    assert seen["thread"] is not threading.main_thread()


async def test_exception_reaches_the_caller():
    def job():
        raise ValueError("API down")

    with pytest.raises(ValueError, match="API down"):
        await run_in_daemon_thread(job)
