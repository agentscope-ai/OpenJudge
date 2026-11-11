"""Exercise callable aggregators through the public grading runner."""

import pytest

from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderScore
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from openjudge.runner.grading_runner import GradingRunner


def copy_score(results):
    """Use the only grader's score as the aggregate."""
    return GraderScore(name="copy_score", score=results["score"].score, reason="Copied score")


@pytest.mark.asyncio
@pytest.mark.parametrize("multiple_datasets", [False, True])
@pytest.mark.parametrize("kind", ["function", "function_list", "base_aggregator", "none"])
async def test_runner_aggregator_forms(kind, multiple_datasets):
    """A single function must work like its one-element-list equivalent."""

    async def evaluate(value):
        return GraderScore(name="score", score=value, reason="Supplied score")

    aggregators = {
        "function": copy_score,
        "function_list": [copy_score],
        "base_aggregator": WeightedSumAggregator(name="weighted_sum", weights={"score": 1.0}),
        "none": None,
    }
    runner = GradingRunner(
        {"score": FunctionGrader(evaluate, name="score")},
        aggregators=aggregators[kind],
        show_progress=False,
    )
    dataset = [{"value": 0.25}, {"value": 0.75}]
    if multiple_datasets:
        runs = await runner.arun_multiple_datasets([dataset, dataset[:1]])
    else:
        runs = [await runner.arun(dataset)]

    for run, expected in zip(runs, [[0.25, 0.75], [0.25]]):
        assert [result.score for result in run["score"]] == expected
        if kind == "none":
            assert set(run) == {"score"}
        else:
            key = "weighted_sum" if kind == "base_aggregator" else "copy_score"
            assert set(run) == {"score", key}
            assert [result.score for result in run[key]] == expected
