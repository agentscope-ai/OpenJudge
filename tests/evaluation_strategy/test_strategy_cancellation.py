# -*- coding: utf-8 -*-
"""Concurrent evaluation strategies own their outstanding comparisons."""

import asyncio

import pytest

from openjudge.evaluation_strategy.average_evaluation_strategy import (
    AverageEvaluationStrategy,
)
from openjudge.evaluation_strategy.grpo_tournament_evaluation_strategy import (
    GRPOTournamentEvaluationStrategy,
)
from openjudge.evaluation_strategy.voting_evaluation_strategy import (
    VotingEvaluationStrategy,
)
from openjudge.graders.schema import GraderRank, GraderScore

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize(
    "strategy",
    [
        AverageEvaluationStrategy(3),
        VotingEvaluationStrategy(3),
        GRPOTournamentEvaluationStrategy(),
        GRPOTournamentEvaluationStrategy(debiased=True),
    ],
)
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_strategy_failure_drains_other_evaluations(strategy, failure) -> None:
    """Errors and cancellation do not leave other repetitions or comparisons running."""
    existing_tasks = asyncio.all_tasks()
    sibling_started = asyncio.Event()
    release = asyncio.Event()
    active = set()
    first = True

    async def evaluate(**kwargs):
        nonlocal first
        if first:
            first = False
            await sibling_started.wait()
            raise failure("evaluation stopped")
        task = asyncio.current_task()
        active.add(task)
        sibling_started.set()
        try:
            await release.wait()
        finally:
            await asyncio.sleep(0.01)
            active.remove(task)

    try:
        with pytest.raises(failure) as raised:
            await asyncio.wait_for(strategy.execute(evaluate, query="q", responses=["a", "b", "c"]), timeout=2)
        # Python 3.10's asyncio.gather drops a child's CancelledError message.
        if failure is RuntimeError:
            assert str(raised.value) == "evaluation stopped"
        assert not active
    finally:
        release.set()
        pending = asyncio.all_tasks() - existing_tasks
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.parametrize(
    "strategy",
    [
        AverageEvaluationStrategy(3),
        VotingEvaluationStrategy(3),
        GRPOTournamentEvaluationStrategy(),
        GRPOTournamentEvaluationStrategy(debiased=True),
    ],
)
async def test_strategies_accept_future_results(strategy) -> None:
    """The evaluation callback may return any Awaitable, including an existing Future."""

    def evaluate(**kwargs):
        result = asyncio.get_running_loop().create_future()
        if "response_1" in kwargs:
            result.set_result(GraderRank(name="local", rank=[1, 2], reason="Local comparison"))
        else:
            result.set_result(GraderScore(name="local", score=1, reason="Local grade"))
        return result

    result = await strategy.execute(evaluate, query="q", responses=["a", "b", "c"])
    if isinstance(strategy, GRPOTournamentEvaluationStrategy):
        assert len(result) == 3
    else:
        assert result.score == 1
