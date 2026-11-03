# -*- coding: utf-8 -*-
"""Cancellation owns active and queued grading work, including progress-enabled runs."""

import asyncio

import pytest

from openjudge.evaluation_strategy.average_evaluation_strategy import (
    AverageEvaluationStrategy,
)
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderScore
from openjudge.runner.grading_runner import GradingRunner

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("show_progress", [False, True])
@pytest.mark.parametrize("multiple_datasets", [False, True])
@pytest.mark.parametrize("repeated_evaluation", [False, True])
async def test_cancellation_drains_active_and_queued_grades(
    show_progress: bool, multiple_datasets: bool, repeated_evaluation: bool
) -> None:
    """Cancellation completes async cleanup and frees permits before the runner is reused."""
    existing_tasks = asyncio.all_tasks()
    two_started = asyncio.Event()
    release = asyncio.Event()
    active = set()
    started = []
    finished = []
    cleaned = []

    async def grade(index: int) -> GraderScore:
        task = asyncio.current_task()
        active.add(task)
        started.append(index)
        if len(active) == 2:
            two_started.set()
        try:
            await release.wait()
            finished.append(index)
            return GraderScore(name="local", score=float(index), reason="Local grade")
        finally:
            await asyncio.sleep(0)
            active.remove(task)
            cleaned.append(index)

    grader = FunctionGrader(grade, name="local", strategy=AverageEvaluationStrategy(2) if repeated_evaluation else None)
    runner = GradingRunner({"local": grader}, max_concurrency=2, show_progress=show_progress)
    dataset = [{"index": i} for i in range(4)]
    run = asyncio.create_task(
        runner.arun_multiple_datasets([dataset, dataset]) if multiple_datasets else runner.arun(dataset)
    )
    try:
        await asyncio.wait_for(two_started.wait(), timeout=2)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, timeout=2)
        assert not active
        assert len(cleaned) == 2
        assert len(started) == 2
        assert not finished
        assert runner.show_progress is show_progress
        release.set()
        result = await asyncio.wait_for(runner.arun([{"index": 9}]), timeout=2)
        assert result["local"][0].score == 9
        assert set(finished) == {9}
    finally:
        release.set()
        pending = asyncio.all_tasks() - existing_tasks
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.parametrize("show_progress", [False, True])
@pytest.mark.parametrize("multiple_datasets", [False, True])
async def test_cancelled_grader_drains_sibling_work(show_progress: bool, multiple_datasets: bool) -> None:
    """An individual cancelled grader does not orphan its siblings or other datasets."""
    existing_tasks = asyncio.all_tasks()
    sibling_started = asyncio.Event()
    release = asyncio.Event()
    active = set()

    async def grade(index: int) -> GraderScore:
        if index == 0:
            await sibling_started.wait()
            raise asyncio.CancelledError()
        task = asyncio.current_task()
        active.add(task)
        sibling_started.set()
        try:
            await release.wait()
            return GraderScore(name="local", score=1, reason="Local grade")
        finally:
            await asyncio.sleep(0)
            active.remove(task)

    runner = GradingRunner({"local": FunctionGrader(grade, name="local")}, show_progress=show_progress)
    dataset = [{"index": 0}, {"index": 1}]
    try:
        with pytest.raises(asyncio.CancelledError):
            if multiple_datasets:
                await asyncio.wait_for(runner.arun_multiple_datasets([dataset, [{"index": 2}]]), timeout=2)
            else:
                await asyncio.wait_for(runner.arun(dataset), timeout=2)
        assert not active
        assert runner.show_progress is show_progress
    finally:
        release.set()
        pending = asyncio.all_tasks() - existing_tasks
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
