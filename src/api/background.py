"""Fire-and-forget background tasks of the API (ingest and eval jobs).

asyncio keeps only a weak reference to a task — without a strong one the GC can cancel a job mid-run.
"""
import asyncio

# Keep strong refs to background tasks so the GC can't cancel them mid-run.
_background_tasks: set[asyncio.Task] = set()


def spawn_background(coro) -> asyncio.Task:
    """Create a background task and retain a strong reference until it completes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
