# -*- coding: utf-8 -*-
"""
Test Grading Runner

Tests for the GradingRunner class functionality.
"""

import copy
from unittest.mock import AsyncMock

import pytest

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderError, GraderScore
from openjudge.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from openjudge.runner.grading_runner import GradingRunner


class MockGrader(BaseGrader):
    """Mock grader for testing purposes."""

    def __init__(self, name="mock_grader", score_value=1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.score_value = score_value
        self.call_args_list = []
        self.call_count = 0

    async def _aevaluate(self, **kwargs):
        """Mock evaluation that returns a fixed score."""
        self.call_count += 1
        # Store a deep copy of the arguments to prevent reference issues
        args_copy = copy.deepcopy(kwargs)
        self.call_args_list.append(args_copy)
        return GraderScore(
            name=self.name,
            score=self.score_value,
            reason=f"Mock score of {self.score_value}",
            metadata={"called_with": args_copy, "call_count": self.call_count},
        )


@pytest.mark.unit
class TestGradingRunner:
    """Test suite for GradingRunner"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_grader1 = MockGrader(name="accuracy_grader")
        mock_grader2 = MockGrader(name="relevance_grader")

        runner = GradingRunner(
            grader_configs={
                "accuracy": mock_grader1,
                "relevance": mock_grader2,
            },
            max_concurrency=2,
        )

        assert runner.max_concurrency == 2
        assert "accuracy" in runner.grader_configs
        assert "relevance" in runner.grader_configs

    @pytest.mark.asyncio
    async def test_grading_runner_with_mock_grader(self):
        """Test the grading runner with mock graders to verify functionality without external dependencies."""

        # Create mock graders
        mock_grader1 = MockGrader(name="accuracy_grader", score_value=0.9)
        mock_grader2 = MockGrader(name="relevance_grader", score_value=0.8)

        # Create runner with mock graders
        runner = GradingRunner(
            grader_configs={
                "accuracy": mock_grader1,
                "relevance": mock_grader2,
            },
            max_concurrency=2,
        )

        # Test data
        dataset = [
            {"query": "What is the capital of France?", "answer": "Paris"},
            {"query": "What is the capital of Germany?", "answer": "Berlin"},
        ]

        # Run the evaluation
        results = await runner.arun(dataset)

        # Verify the structure of results
        assert "accuracy" in results
        assert "relevance" in results
        assert len(results["accuracy"]) == 2
        assert len(results["relevance"]) == 2

        # Verify that each result is a GraderScore
        for grader_results in results.values():
            for result in grader_results:
                assert isinstance(result, (GraderScore, GraderError))

        # Verify specific scores
        assert results["accuracy"][0].score == 0.9
        assert results["relevance"][1].score == 0.8

        # Verify that graders were called isolatedly
        for name, _results in results.items():
            for result in _results:
                assert result.metadata["call_count"] == 1

    @pytest.mark.asyncio
    async def test_grading_runner_with_mappers(self):
        """Test the grading runner with data mappers integrated in graders."""

        # Create mock grader with mapper integrated
        # The mapper format is {target_field: source_field_in_original_data}
        mock_grader = MockGrader(
            name="mapped_grader",
        )

        # Create runner with the grader that has integrated mapper
        runner = GradingRunner(
            grader_configs={
                "mapped_test": (mock_grader, {"query": "question", "answer": "response"}),
            },
        )

        # Test data with different field names
        dataset = [
            {"question": "What is 2+2?", "resp": "4"},
            {"question": "What is the sky color?", "resp": "blue"},
        ]

        # Run the evaluation
        results = await runner.arun(dataset)

        # Verify results structure
        assert "mapped_test" in results
        assert len(results["mapped_test"]) == 2

    @pytest.mark.asyncio
    async def test_grading_runner_with_aggregators(self):
        """Test the grading runner with aggregators."""

        # Create mock graders
        mock_grader1 = MockGrader(name="accuracy_grader", score_value=0.9)
        mock_grader2 = MockGrader(name="relevance_grader", score_value=0.8)

        # Create aggregator
        aggregator = WeightedSumAggregator(
            name="weighted_sum",
            weights={"accuracy_grader": 0.6, "relevance_grader": 0.4},
        )

        # Create runner with aggregators
        runner = GradingRunner(
            grader_configs={
                "accuracy_grader": mock_grader1,
                "relevance_grader": mock_grader2,
            },
            aggregators=[aggregator],
        )

        # Test data
        dataset = [
            {"query": "What is the capital of France?", "answer": "Paris"},
        ]

        # Run the evaluation
        results = await runner.arun(dataset)

        # Verify the structure of results
        assert "accuracy_grader" in results
        assert "relevance_grader" in results

        # Find the aggregator results - they are stored with the aggregator name as key
        assert "weighted_sum" in results
        aggregator_results = results["weighted_sum"]

        assert aggregator_results is not None
        assert len(aggregator_results) == 1

        # Verify that each result is a GraderScore
        for grader_results in results.values():
            for result in grader_results:
                assert isinstance(result, (GraderScore, GraderError))

        # Verify aggregated score calculation
        # Expected: (0.9 * 0.6 + 0.8 * 0.4) / (0.6 + 0.4) = (0.54 + 0.32) / 1.0 = 0.86
        assert aggregator_results[0].score == pytest.approx(0.86)

    @pytest.mark.asyncio
    async def test_grading_runner_error_handling(self):
        """Test the grading runner error handling."""

        class ErrorGrader(BaseGrader):
            def __init__(self, name="error_grader", mapper=None):
                super().__init__(name=name, mapper=mapper)

            async def _aevaluate(self, **kwargs):
                raise Exception("Test error")

        # Create mock graders - one normal, one that raises error
        mock_grader = MockGrader(name="normal_grader", score_value=0.9)
        error_grader = ErrorGrader(name="error_grader")

        # Create runner
        runner = GradingRunner(
            grader_configs={
                "normal": mock_grader,
                "error": error_grader,
            },
        )

        # Test data
        dataset = [
            {"query": "What is the capital of France?", "answer": "Paris"},
        ]

        # Run the evaluation
        results = await runner.arun(dataset)

        # Verify the structure of results
        assert "normal" in results
        assert "error" in results
        assert len(results["normal"]) == 1
        assert len(results["error"]) == 1

        # Normal grader should return GraderScore
        assert isinstance(results["normal"][0], GraderScore)
        assert results["normal"][0].score == 0.9

        # Error grader should return GraderError
        assert isinstance(results["error"][0], GraderError)
        assert "Error" in results["error"][0].error

    @pytest.mark.asyncio
    async def test_grading_runner_with_real_components(self):
        """Test the grading runner with real components but mocked models."""

        # Create a mock model
        mock_model = AsyncMock()
        mock_response = AsyncMock()
        mock_response.score = 5
        mock_response.reason = "Excellent"

        mock_response.parsed = {}
        mock_model.achat = AsyncMock(return_value=mock_response)

        # We won't actually instantiate real graders that require API keys,
        # but we can test the runner structure with mock components
        mock_grader1 = MockGrader(name="accuracy_grader", score_value=0.9)
        mock_grader2 = MockGrader(name="relevance_grader", score_value=0.85)

        # Create runner
        runner = GradingRunner(
            grader_configs={
                "accuracy": mock_grader1,
                "relevance": mock_grader2,
            },
            max_concurrency=2,
        )

        # Test data
        dataset = [
            {"query": "What is the capital of France?", "answer": "Paris"},
            {"query": "What is the capital of Germany?", "answer": "Berlin"},
            {"query": "What is the capital of Italy?", "answer": "Rome"},
        ]

        # Run the evaluation
        results = await runner.arun(dataset)

        # Verify the structure of results
        assert "accuracy" in results
        assert "relevance" in results
        assert len(results["accuracy"]) == 3
        assert len(results["relevance"]) == 3

        # Verify that each result is a GraderScore
        for grader_results in results.values():
            for result in grader_results:
                assert isinstance(result, (GraderScore, GraderError))
                assert result.metadata["call_count"] == 1

    @pytest.mark.asyncio
    async def test_grading_runner_multiple_datasets(self):
        """Test the grading runner with multiple datasets using arun_multiple_datasets."""

        # Create mock graders
        mock_grader1 = MockGrader(name="accuracy_grader", score_value=0.9)
        mock_grader2 = MockGrader(name="relevance_grader", score_value=0.8)

        # Create runner with mock graders
        runner = GradingRunner(
            grader_configs={
                "accuracy": mock_grader1,
                "relevance": mock_grader2,
            },
            max_concurrency=5,
            show_progress=False,  # Disable progress bar for testing
        )

        # Test data - multiple datasets
        datasets = [
            [  # dataset_0
                {"query": "What is the capital of France?", "answer": "Paris"},
                {"query": "What is the capital of Germany?", "answer": "Berlin"},
            ],
            [  # dataset_1
                {"query": "What is 2+2?", "answer": "4"},
            ],
            [  # dataset_2
                {"query": "What is the sky color?", "answer": "blue"},
                {"query": "What is the grass color?", "answer": "green"},
                {"query": "What is the sun color?", "answer": "yellow"},
            ],
        ]

        # Run multiple_datasets evaluation
        results = await runner.arun_multiple_datasets(datasets)

        # Verify the structure of results - should be a list
        assert isinstance(results, list)
        assert len(results) == 3

        # Verify dataset_0 results
        assert "accuracy" in results[0]
        assert "relevance" in results[0]
        assert len(results[0]["accuracy"]) == 2
        assert len(results[0]["relevance"]) == 2

        # Verify dataset_1 results
        assert len(results[1]["accuracy"]) == 1
        assert len(results[1]["relevance"]) == 1

        # Verify dataset_2 results
        assert len(results[2]["accuracy"]) == 3
        assert len(results[2]["relevance"]) == 3

        # Verify that all results are GraderScore instances
        for dataset_results in results:
            for grader_name, grader_results in dataset_results.items():
                for result in grader_results:
                    assert isinstance(result, (GraderScore, GraderError))
                    if isinstance(result, GraderScore):
                        # Verify scores match expected values
                        if grader_name == "accuracy":
                            assert result.score == 0.9
                        elif grader_name == "relevance":
                            assert result.score == 0.8

    @pytest.mark.asyncio
    async def test_grading_runner_multiple_datasets_with_empty_dataset(self):
        """Test the grading runner multiple_datasets with an empty dataset."""

        # Create mock graders
        mock_grader = MockGrader(name="test_grader", score_value=0.9)

        # Create runner
        runner = GradingRunner(
            grader_configs={"test": mock_grader},
            show_progress=False,
        )

        # Test data with one empty dataset
        datasets = [
            [{"query": "Q1", "answer": "A1"}],
            [],  # Empty dataset
            [{"query": "Q2", "answer": "A2"}],
        ]

        # Run multiple_datasets evaluation
        results = await runner.arun_multiple_datasets(datasets)

        # Verify structure - should be a list
        assert isinstance(results, list)
        assert len(results) == 3

        # Verify dataset_0 has 1 result
        assert len(results[0]["test"]) == 1

        # Verify dataset_1 is empty
        assert len(results[1]["test"]) == 0

        # Verify dataset_2 has 1 result
        assert len(results[2]["test"]) == 1

    @pytest.mark.asyncio
    async def test_grading_runner_multiple_datasets_with_aggregators(self):
        """Test the grading runner multiple_datasets with aggregators."""

        # Create mock graders
        mock_grader1 = MockGrader(name="accuracy_grader", score_value=0.9)
        mock_grader2 = MockGrader(name="relevance_grader", score_value=0.8)

        # Create aggregator
        aggregator = WeightedSumAggregator(
            name="weighted_sum",
            weights={"accuracy_grader": 0.6, "relevance_grader": 0.4},
        )

        # Create runner with aggregators
        runner = GradingRunner(
            grader_configs={
                "accuracy_grader": mock_grader1,
                "relevance_grader": mock_grader2,
            },
            aggregators=[aggregator],
            show_progress=False,
        )

        # Test data
        datasets = [
            [{"query": "Q1", "answer": "A1"}],
            [{"query": "Q2", "answer": "A2"}, {"query": "Q3", "answer": "A3"}],
        ]

        # Run multiple_datasets evaluation
        results = await runner.arun_multiple_datasets(datasets)

        # Verify structure - should be a list
        assert isinstance(results, list)
        assert len(results) == 2

        # Verify aggregator results exist in both datasets
        for dataset_results in results:
            # Find aggregator results - they are stored with the aggregator name as key
            assert "weighted_sum" in dataset_results
            aggregator_results = dataset_results["weighted_sum"]

            assert aggregator_results is not None

            # Verify aggregated scores
            for result in aggregator_results:
                assert isinstance(result, GraderScore)
                # Expected: (0.9 * 0.6 + 0.8 * 0.4) / (0.6 + 0.4) = 0.86
                assert result.score == pytest.approx(0.86)

    @pytest.mark.asyncio
    async def test_grading_runner_multiple_datasets_concurrency_sharing(self):
        """Test that all datasets share the same concurrency pool."""

        # Create a mock grader that tracks execution timing
        import asyncio
        import time

        class TimingMockGrader(BaseGrader):
            def __init__(self, name="timing_grader", delay=0.1, mapper=None):
                super().__init__(name=name, mapper=mapper)
                self.delay = delay
                self.execution_times = []

            async def _aevaluate(self, **kwargs):
                start_time = time.time()
                await asyncio.sleep(self.delay)
                end_time = time.time()
                self.execution_times.append((start_time, end_time))
                return GraderScore(
                    name=self.name,
                    score=1.0,
                    reason="Test",
                )

        # Create grader with small delay
        timing_grader = TimingMockGrader(name="timing_grader", delay=0.05)

        # Create runner with low concurrency
        runner = GradingRunner(
            grader_configs={"timing": timing_grader},
            max_concurrency=2,  # Only 2 concurrent tasks
            show_progress=False,
        )

        # Create multiple datasets with multiple samples
        datasets = [
            [{"query": f"Q{i}", "answer": f"A{i}"} for i in range(3)],
            [{"query": f"Q{i}", "answer": f"A{i}"} for i in range(3, 6)],
        ]

        # Run multiple_datasets evaluation
        start = time.time()
        results = await runner.arun_multiple_datasets(datasets)
        end = time.time()

        # Verify results - should be a list
        assert isinstance(results, list)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_grading_runner_multiple_datasets_order_preservation(self):
        """Test that arun_multiple_datasets preserves order at all three levels: dataset, grader, and sample."""

        # Create a mock grader that returns the input data in the score
        class OrderTrackingGrader(BaseGrader):
            def __init__(self, name="order_grader", mapper=None):
                super().__init__(name=name, mapper=mapper)
                self.call_order = []

            async def _aevaluate(self, **kwargs):
                # Record the order of calls
                self.call_order.append(kwargs)
                # Return a score that includes the input data for verification
                return GraderScore(
                    name=self.name,
                    score=1.0,
                    reason=f"Processed {kwargs.get('query', 'unknown')}",
                    metadata=kwargs,
                )

        # Create two graders to test grader-level ordering
        grader1 = OrderTrackingGrader(name="grader_1")
        grader2 = OrderTrackingGrader(name="grader_2")

        # Create runner
        runner = GradingRunner(
            grader_configs={
                "grader_1": grader1,
                "grader_2": grader2,
            },
            max_concurrency=10,  # High concurrency to stress test ordering
            show_progress=False,
        )

        # Create datasets with identifiable data
        datasets = [
            [  # dataset_0
                {"query": "D0_S0", "answer": "A0"},
                {"query": "D0_S1", "answer": "A1"},
                {"query": "D0_S2", "answer": "A2"},
            ],
            [  # dataset_1
                {"query": "D1_S0", "answer": "B0"},
                {"query": "D1_S1", "answer": "B1"},
            ],
            [  # dataset_2
                {"query": "D2_S0", "answer": "C0"},
                {"query": "D2_S1", "answer": "C1"},
                {"query": "D2_S2", "answer": "C2"},
                {"query": "D2_S3", "answer": "C3"},
            ],
        ]

        # Run multiple_datasets evaluation
        results = await runner.arun_multiple_datasets(datasets)

        # === Level 1: Verify dataset-level ordering ===
        assert isinstance(results, list)
        assert len(results) == 3, "Should have 3 datasets"

        # === Level 2: Verify grader-level ordering within each dataset ===
        for i, dataset_results in enumerate(results):
            grader_keys = [k for k in dataset_results.keys() if isinstance(k, str)]
            assert "grader_1" in grader_keys, f"grader_1 should be in dataset {i}"
            assert "grader_2" in grader_keys, f"grader_2 should be in dataset {i}"

        # === Level 3: Verify sample-level ordering within each grader ===
        # Check dataset_0
        assert len(results[0]["grader_1"]) == 3
        assert results[0]["grader_1"][0].metadata["query"] == "D0_S0"
        assert results[0]["grader_1"][1].metadata["query"] == "D0_S1"
        assert results[0]["grader_1"][2].metadata["query"] == "D0_S2"

        assert len(results[0]["grader_2"]) == 3
        assert results[0]["grader_2"][0].metadata["query"] == "D0_S0"
        assert results[0]["grader_2"][1].metadata["query"] == "D0_S1"
        assert results[0]["grader_2"][2].metadata["query"] == "D0_S2"

        # Check dataset_1
        assert len(results[1]["grader_1"]) == 2
        assert results[1]["grader_1"][0].metadata["query"] == "D1_S0"
        assert results[1]["grader_1"][1].metadata["query"] == "D1_S1"

        assert len(results[1]["grader_2"]) == 2
        assert results[1]["grader_2"][0].metadata["query"] == "D1_S0"
        assert results[1]["grader_2"][1].metadata["query"] == "D1_S1"

        # Check dataset_2
        assert len(results[2]["grader_1"]) == 4
        assert results[2]["grader_1"][0].metadata["query"] == "D2_S0"
        assert results[2]["grader_1"][1].metadata["query"] == "D2_S1"
        assert results[2]["grader_1"][2].metadata["query"] == "D2_S2"
        assert results[2]["grader_1"][3].metadata["query"] == "D2_S3"

        assert len(results[2]["grader_2"]) == 4
        assert results[2]["grader_2"][0].metadata["query"] == "D2_S0"
        assert results[2]["grader_2"][1].metadata["query"] == "D2_S1"
        assert results[2]["grader_2"][2].metadata["query"] == "D2_S2"
        assert results[2]["grader_2"][3].metadata["query"] == "D2_S3"

        # === Verify answer ordering matches query ordering ===
        for i, (dataset_results, dataset) in enumerate(zip(results, datasets)):
            for grader_name in ["grader_1", "grader_2"]:
                grader_results = dataset_results[grader_name]
                for j, (result, expected_sample) in enumerate(zip(grader_results, dataset)):
                    assert (
                        result.metadata["query"] == expected_sample["query"]
                    ), f"Query mismatch at dataset {i}, {grader_name}, sample {j}"
                    assert (
                        result.metadata["answer"] == expected_sample["answer"]
                    ), f"Answer mismatch at dataset {i}, {grader_name}, sample {j}"

    @pytest.mark.asyncio
    async def test_grading_runner_multiple_datasets_order_under_high_concurrency(self):
        """Test that order is preserved even with random delays simulating real-world async behavior."""
        import asyncio
        import random

        class RandomDelayGrader(BaseGrader):
            def __init__(self, name="random_grader", mapper=None):
                super().__init__(name=name, mapper=mapper)

            async def _aevaluate(self, **kwargs):
                # Random delay to simulate varying processing times
                await asyncio.sleep(random.uniform(0.001, 0.01))
                return GraderScore(
                    name=self.name,
                    score=1.0,
                    reason="Test",
                    metadata=kwargs,
                )

        # Create grader
        grader = RandomDelayGrader(name="random_grader")

        # Create runner with high concurrency
        runner = GradingRunner(
            grader_configs={"random_grader": grader},
            max_concurrency=20,
            show_progress=False,
        )

        # Create multiple datasets with many samples
        datasets = [[{"query": f"D{d}_S{s}", "answer": f"A{d}_{s}"} for s in range(10)] for d in range(5)]

        # Run multiple_datasets evaluation multiple times to test consistency
        for run in range(3):
            results = await runner.arun_multiple_datasets(datasets)

            # Verify all datasets are present and in order
            assert isinstance(results, list)
            assert len(results) == 5

            # Verify sample order in each dataset
            for d in range(5):
                grader_results = results[d]["random_grader"]
                assert len(grader_results) == 10

                # Check that samples are in the correct order
                for s in range(10):
                    expected_query = f"D{d}_S{s}"
                    expected_answer = f"A{d}_{s}"
                    assert (
                        grader_results[s].metadata["query"] == expected_query
                    ), f"Run {run}: Query order mismatch at dataset {d}, sample {s}"
                    assert (
                        grader_results[s].metadata["answer"] == expected_answer
                    ), f"Run {run}: Answer order mismatch at dataset {d}, sample {s}"

    @pytest.mark.asyncio
    async def test_grading_runner_multiple_datasets_order_with_errors(self):
        """Test that order is preserved even when some evaluations fail."""

        class SelectiveErrorGrader(BaseGrader):
            def __init__(self, name="error_grader", error_queries=None, mapper=None):
                super().__init__(name=name, mapper=mapper)
                self.error_queries = error_queries or []

            async def _aevaluate(self, **kwargs):
                query = kwargs.get("query", "")
                if query in self.error_queries:
                    raise ValueError(f"Intentional error for {query}")
                return GraderScore(
                    name=self.name,
                    score=1.0,
                    reason="Success",
                    metadata=kwargs,
                )

        # Create grader that will fail on specific queries
        grader = SelectiveErrorGrader(name="selective_grader", error_queries=["D0_S1", "D1_S0", "D2_S2"])

        # Create runner
        runner = GradingRunner(
            grader_configs={"selective_grader": grader},
            max_concurrency=10,
            show_progress=False,
        )

        # Create datasets
        datasets = [
            [  # dataset_0
                {"query": "D0_S0", "answer": "A0"},
                {"query": "D0_S1", "answer": "A1"},  # Will error
                {"query": "D0_S2", "answer": "A2"},
            ],
            [  # dataset_1
                {"query": "D1_S0", "answer": "B0"},  # Will error
                {"query": "D1_S1", "answer": "B1"},
            ],
            [  # dataset_2
                {"query": "D2_S0", "answer": "C0"},
                {"query": "D2_S1", "answer": "C1"},
                {"query": "D2_S2", "answer": "C2"},  # Will error
            ],
        ]

        # Run multiple_datasets evaluation
        results = await runner.arun_multiple_datasets(datasets)

        # Verify dataset order
        assert isinstance(results, list)
        assert len(results) == 3

        # Verify dataset_0: success, error, success
        assert len(results[0]["selective_grader"]) == 3
        assert isinstance(results[0]["selective_grader"][0], GraderScore)
        assert results[0]["selective_grader"][0].metadata["query"] == "D0_S0"

        assert isinstance(results[0]["selective_grader"][1], GraderError)
        assert "D0_S1" in results[0]["selective_grader"][1].error

        assert isinstance(results[0]["selective_grader"][2], GraderScore)
        assert results[0]["selective_grader"][2].metadata["query"] == "D0_S2"

        # Verify dataset_1: error, success
        assert len(results[1]["selective_grader"]) == 2
        assert isinstance(results[1]["selective_grader"][0], GraderError)
        assert "D1_S0" in results[1]["selective_grader"][0].error

        assert isinstance(results[1]["selective_grader"][1], GraderScore)
        assert results[1]["selective_grader"][1].metadata["query"] == "D1_S1"

        # Verify dataset_2: success, success, error
        assert len(results[2]["selective_grader"]) == 3
        assert isinstance(results[2]["selective_grader"][0], GraderScore)
        assert results[2]["selective_grader"][0].metadata["query"] == "D2_S0"

        assert isinstance(results[2]["selective_grader"][1], GraderScore)
        assert results[2]["selective_grader"][1].metadata["query"] == "D2_S1"

        assert isinstance(results[2]["selective_grader"][2], GraderError)
        assert "D2_S2" in results[2]["selective_grader"][2].error
