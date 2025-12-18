# -*- coding: utf-8 -*-
"""
Test Grading Runner

Tests for the GradingRunner class functionality.
"""

import copy
from unittest.mock import AsyncMock

import pytest
from jsonschema import validate

from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderError, GraderScore
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.runner.aggregator.weighted_sum_aggregator import (
    WeightedSumAggregator,
)
from rm_gallery.core.runner.grading_runner import GradingRunner


class MockGrader(BaseGrader):
    """Mock grader for testing purposes."""

    def __init__(self, name="mock_grader", score_value=1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.score_value = score_value
        self.call_args_list = []
        self.call_count = 0

    async def aevaluate(self, **kwargs):
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

        # Verify that graders were called correctly
        assert mock_grader1.call_count == 2
        assert mock_grader2.call_count == 2

        # Verify the data passed to graders - we now check all calls for both graders
        # Since the execution is concurrent, we can't guarantee the order of calls
        all_calls_accuracy = [call for call in mock_grader1.call_args_list]
        all_calls_relevance = [call for call in mock_grader2.call_args_list]

        # Check that each grader was called with both data points
        accuracy_queries = [call["query"] for call in all_calls_accuracy]
        relevance_queries = [call["query"] for call in all_calls_relevance]

        assert "What is the capital of France?" in accuracy_queries
        assert "What is the capital of Germany?" in accuracy_queries
        assert "What is the capital of France?" in relevance_queries
        assert "What is the capital of Germany?" in relevance_queries

        # Check that each grader was called with corresponding answers
        france_call_idx = accuracy_queries.index("What is the capital of France?")
        germany_call_idx = accuracy_queries.index("What is the capital of Germany?")

        assert all_calls_accuracy[france_call_idx]["answer"] == "Paris"
        assert all_calls_accuracy[germany_call_idx]["answer"] == "Berlin"

    @pytest.mark.asyncio
    async def test_grading_runner_with_mappers(self):
        """Test the grading runner with data mappers."""

        # Create mock grader
        mock_grader = MockGrader(name="mapped_grader")

        # Create runner with mapper
        # The mapper format is {new_field_name: path_in_original_data}
        runner = GradingRunner(
            grader_configs={
                "mapped_test": (mock_grader, {"query": "question", "answer": "response"}),
            },
        )

        # Test data with different field names
        dataset = [
            {"question": "What is 2+2?", "response": "4"},
            {"question": "What is the sky color?", "response": "blue"},
        ]

        # Run the evaluation
        results = await runner.arun(dataset)

        # Verify results structure
        assert "mapped_test" in results
        assert len(results["mapped_test"]) == 2

        # Verify that the mapper worked - the grader should have been called with mapped field names
        assert len(mock_grader.call_args_list) == 2
        all_calls = [call for call in mock_grader.call_args_list]

        # Check that each call has the mapped field names
        for call in all_calls:
            assert "query" in call
            assert "answer" in call

        # Check that the data was correctly mapped
        queries = [call["query"] for call in all_calls]
        answers = [call["answer"] for call in all_calls]

        assert "What is 2+2?" in queries
        assert "What is the sky color?" in queries
        assert "4" in answers
        assert "blue" in answers

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

        # Find the aggregator results - they are stored with the aggregator object as key
        aggregator_results = None
        for key in results:
            if hasattr(key, "__call__"):  # It's the aggregator object
                aggregator_results = results[key]
                break

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
            def __init__(self, name="error_grader"):
                super().__init__(name=name)

            async def aevaluate(self, **kwargs):
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

        # Verify all graders were called the right number of times
        assert mock_grader1.call_count == 3
        assert mock_grader2.call_count == 3
