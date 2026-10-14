"""Fire-and-forget background tasks of the API (ingest and eval jobs).

asyncio keeps only a weak reference to a task — without a strong one the GC can cancel a job mid-run.
"""
import asyncio
import threading
from collections.abc import Callable
from typing import Any

# Keep strong refs to background tasks so the GC can't cancel them mid-run.
_background_tasks: set[asyncio.Task] = set()


def spawn_background(coro) -> asyncio.Task:
    """Create a background task and retain a strong reference until it completes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def run_in_daemon_thread(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a blocking ``fn(*args)`` in a daemon thread and await its result (or exception).

    For jobs that may be abandoned when the process ends. ``asyncio.to_thread`` uses the default executor,
    whose threads Python joins at exit: a running end-to-end eval kept a stopped dev worker alive, and
    uvicorn's reloader waits for the old worker without a limit — no new worker started (roadmap 12.3).
    A daemon thread dies with the process instead. Not for jobs that write (ingest): those must finish.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def deliver(setter: Callable[[Any], None], value: Any) -> None:
        if not future.done():  # the awaiting task may have been cancelled meanwhile
            setter(value)

    def runner() -> None:
        try:
            result = fn(*args)
        except BaseException as e:  # handed to the awaiting coroutine, which decides
            outcome: tuple[Callable[[Any], None], Any] = (future.set_exception, e)
        else:
            outcome = (future.set_result, result)
        try:
            loop.call_soon_threadsafe(deliver, *outcome)
        except RuntimeError:
            pass  # the loop is already closed — the process is ending, nobody waits for the result

    threading.Thread(target=runner, name=f"job-{getattr(fn, '__name__', 'fn')}", daemon=True).start()
    return await future
