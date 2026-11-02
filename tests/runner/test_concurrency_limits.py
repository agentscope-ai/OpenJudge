# -*- coding: utf-8 -*-
"""Tests for positive concurrency limits and fail-fast configuration."""

import asyncio
from typing import Any, Callable

import pytest

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderScore
from openjudge.runner.grading_runner import GradingRunner
from openjudge.runner.resource_executor import SemaphoreResourceExecutor
from openjudge.utils.concurrency import ConcurrencyManager

pytestmark = pytest.mark.unit


class LocalGrader(BaseGrader):
    """Return a deterministic score without a model or network dependency."""

    async def _aevaluate(self, **kwargs: Any) -> GraderScore:
        return GraderScore(name=self.name, score=1.0, reason="Local deterministic grade")


def make_runner(limit: int) -> GradingRunner:
    """Construct a runner with its default semaphore executor."""
    return GradingRunner({"local": LocalGrader(name="local")}, max_concurrency=limit, show_progress=False)


@pytest.mark.parametrize("factory", [SemaphoreResourceExecutor, ConcurrencyManager, make_runner])
@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_concurrency_is_rejected(factory: Callable[[int], object], limit: int) -> None:
    """An invalid limit fails before any task can wait for a nonexistent permit."""
    with pytest.raises(ValueError, match="max_concurrency must be greater than 0"):
        factory(limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1])
async def test_invalid_limit_update_keeps_the_manager_usable(limit: int) -> None:
    """Reject a bad update without replacing the working semaphore."""
    manager = ConcurrencyManager(1)
    with pytest.raises(ValueError, match="max_concurrency must be greater than 0"):
        manager.set_max_concurrency(limit)

    async def operation() -> str:
        return "done"

    assert await asyncio.wait_for(manager.run_with_concurrency_control(operation()), timeout=1) == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("component", ["executor", "manager"])
async def test_positive_limits_still_bound_running_tasks(component: str) -> None:
    """Real operations observe the configured limit while queued work completes."""
    executor = SemaphoreResourceExecutor(2)
    manager = ConcurrencyManager(1)
    manager.set_max_concurrency(2)
    release = asyncio.Event()
    two_running = asyncio.Event()
    active = 0
    peak = 0

    async def operation(index: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            two_running.set()
        try:
            await release.wait()
            return index
        finally:
            active -= 1

    async def submit(index: int) -> int:
        if component == "executor":
            return await executor.submit(operation, index=index)
        return await manager.run_with_concurrency_control(operation(index))

    tasks = [asyncio.create_task(submit(index)) for index in range(4)]
    try:
        await asyncio.wait_for(two_running.wait(), timeout=1)
        assert active == 2
        release.set()
        assert await asyncio.wait_for(asyncio.gather(*tasks), timeout=1) == [0, 1, 2, 3]
        assert peak == 2
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_runner_with_positive_limit_completes_real_grading() -> None:
    """The public runner processes samples with a working default executor."""
    result = await asyncio.wait_for(make_runner(1).arun([{"query": "one"}, {"query": "two"}]), timeout=1)
    assert [grade.score for grade in result["local"]] == [1.0, 1.0]


@pytest.mark.asyncio
async def test_custom_executor_controls_its_own_capacity() -> None:
    """A custom executor still makes the runner's unused default limit irrelevant."""
    runner = GradingRunner(
        {"local": LocalGrader(name="local")},
        max_concurrency=0,
        executor=SemaphoreResourceExecutor(1),
        show_progress=False,
    )
    result = await asyncio.wait_for(runner.arun([{"query": "local"}]), timeout=1)
    assert result["local"][0].score == 1.0
