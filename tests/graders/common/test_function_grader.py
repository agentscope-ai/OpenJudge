# -*- coding: utf-8 -*-
"""
Tests for FunctionGrader following the GRADER_TESTING_STRATEGY.md guidelines.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using FunctionGrader as an example of Rule-Based Grader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against gold standard datasets)

Since FunctionGrader is Rule-Based, it does not require consistency quality tests.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/common/test_function_grader.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/common/test_function_grader.py -m unit
    ```
"""

import asyncio

import pytest

from openjudge.analyzer.validation import (
    AccuracyAnalyzer,
    CorrelationAnalyzer,
    F1ScoreAnalyzer,
    PrecisionAnalyzer,
    RecallAnalyzer,
)
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderMode, GraderRank, GraderScore
from openjudge.runner.grading_runner import GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation


@pytest.mark.unit
class TestFunctionGraderUnit:
    """Unit tests for FunctionGrader - testing isolated functionality"""

    def test_pointwise_initialization(self):
        """Test successful initialization for pointwise mode"""

        def simple_function(query: str, response: str) -> GraderScore:
            return GraderScore(name="test_pointwise", score=1.0, reason="Test")

        grader = FunctionGrader(
            func=simple_function,
            name="test_pointwise",
            mode=GraderMode.POINTWISE,
            description="Pointwise test function",
        )
        assert grader.name == "test_pointwise"
        assert grader.mode == GraderMode.POINTWISE
        assert grader.description == "Pointwise test function"
        assert grader.func == simple_function

    def test_listwise_initialization(self):
        """Test successful initialization for listwise mode"""

        def ranking_function(query: str, response_1: str, response_2: str) -> GraderRank:
            return GraderRank(name="test_listwise", rank=[1, 2], reason="Test")

        grader = FunctionGrader(
            func=ranking_function,
            name="test_listwise",
            mode=GraderMode.LISTWISE,
            description="Listwise test function",
        )
        assert grader.name == "test_listwise"
        assert grader.mode == GraderMode.LISTWISE
        assert grader.description == "Listwise test function"
        assert grader.func == ranking_function

    @pytest.mark.asyncio
    async def test_async_pointwise_evaluation(self):
        """Test successful async pointwise evaluation with valid inputs"""

        async def async_scoring_function(query: str, response: str) -> GraderScore:
            # Simulate some async work
            await asyncio.sleep(0.01)
            return GraderScore(
                name="test_async_pointwise",
                score=4.5,
                reason="Async evaluation successful",
            )

        grader = FunctionGrader(
            func=async_scoring_function,
            name="test_async_pointwise",
            mode=GraderMode.POINTWISE,
        )

        result = await grader.aevaluate(
            query="What is Python?",
            response="Python is a high-level programming language.",
        )

        assert isinstance(result, GraderScore)
        assert result.name == "test_async_pointwise"
        assert result.score == 4.5
        assert result.reason == "Async evaluation successful"

    @pytest.mark.asyncio
    async def test_sync_pointwise_evaluation(self):
        """Test successful sync pointwise evaluation with valid inputs"""

        def scoring_function(query: str, response: str) -> GraderScore:
            return GraderScore(
                name="test_sync_pointwise",
                score=3.0,
                reason="Sync evaluation successful",
            )

        grader = FunctionGrader(
            func=scoring_function,
            name="test_sync_pointwise",
            mode=GraderMode.POINTWISE,
        )

        result = await grader.aevaluate(
            query="What is Python?",
            response="Python is a high-level programming language.",
        )

        assert isinstance(result, GraderScore)
        assert result.name == "test_sync_pointwise"
        assert result.score == 3.0
        assert result.reason == "Sync evaluation successful"

    @pytest.mark.asyncio
    async def test_listwise_evaluation(self):
        """Test successful listwise evaluation with valid inputs"""

        def ranking_function(query: str, response_1: str, response_2: str) -> GraderRank:
            # Simple ranking based on length
            if len(response_1) > len(response_2):
                return GraderRank(name="test_listwise", rank=[1, 2], reason="First response is longer")
            else:
                return GraderRank(name="test_listwise", rank=[2, 1], reason="Second response is longer")

        grader = FunctionGrader(
            func=ranking_function,
            name="test_listwise",
            mode=GraderMode.LISTWISE,
        )

        result = await grader.aevaluate(
            query="Explain machine learning",
            response_1="Machine learning is a subset of AI.",
            response_2="Machine learning is a method of data analysis that automates analytical model building.",
        )

        assert isinstance(result, GraderRank)
        assert result.name == "test_listwise"
        assert result.rank == [2, 1]
        assert result.reason == "Second response is longer"

    @pytest.mark.asyncio
    async def test_pointwise_type_error_handling(self):
        """Test error handling when pointwise function returns wrong type"""

        def wrong_return_function(query: str, response: str) -> GraderRank:
            # This function returns GraderRank but grader is in POINTWISE mode
            return GraderRank(name="test_pointwise", rank=[1, 2], reason="Wrong return type")

        grader = FunctionGrader(
            func=wrong_return_function,
            name="test_wrong_type",
            mode=GraderMode.POINTWISE,
        )

        # Wrap the call in try/except to catch the actual error
        try:
            await grader.aevaluate(
                query="Test query",
                response="Test response",
            )
            assert False, "Should have raised TypeError"
        except TypeError as e:
            assert "Expected GraderScore for pointwise mode" in str(e)

    @pytest.mark.asyncio
    async def test_listwise_type_error_handling(self):
        """Test error handling when listwise function returns wrong type"""

        def wrong_return_function(query: str, response_1: str, response_2: str) -> GraderScore:
            # This function returns GraderScore but grader is in LISTWISE mode
            return GraderScore(name="test_listwise", score=5.0, reason="Wrong return type")

        grader = FunctionGrader(
            func=wrong_return_function,
            name="test_wrong_type",
            mode=GraderMode.LISTWISE,
        )

        # Wrap the call in try/except to catch the actual error
        try:
            await grader.aevaluate(
                query="Test query",
                response_1="Test response 1",
                response_2="Test response 2",
            )
            assert False, "Should have raised TypeError"
        except TypeError as e:
            assert "Expected GraderRank for listwise mode" in str(e)


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Since FunctionGrader is Rule-Based, it does not require API keys or online testing
# We'll demonstrate quality tests using a predefined function with a gold standard dataset


@pytest.mark.quality
class TestFunctionGraderQuality:
    """Quality tests for FunctionGrader - testing evaluation quality"""

    @pytest.fixture
    def length_based_grader(self):
        """Create a length-based grader for testing"""

        def length_scorer(query: str, response: str) -> GraderScore:
            # Simple scoring function based on response length
            # Longer responses get higher scores, up to a max of 10
            length_score = min(len(response) / 20.0, 10.0)
            return GraderScore(
                name="length_scorer",
                score=length_score,
                reason=f"Response length: {len(response)} characters",
            )

        return FunctionGrader(
            func=length_scorer,
            name="length_scorer",
            mode=GraderMode.POINTWISE,
        )

    @pytest.fixture
    def dataset(self):
        """Load gold standard dataset"""
        return [
            {
                "query": "Explain quantum computing in simple terms",
                "response": "Quantum computing uses quantum bits.",  # Short response
                "human_score": 3,  # Low quality due to brevity
            },
            {
                "query": "Explain quantum computing in simple terms",
                "response": "Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously. "
                + "This allows quantum computers to perform many calculations at once, potentially solving certain "
                + "problems much faster than classical computers.",  # Long, detailed response
                "human_score": 8,  # High quality due to detail
            },
            {
                "query": "What is the weather today?",
                "response": "It's sunny.",  # Very short response
                "human_score": 2,  # Low quality
            },
        ]

    @pytest.mark.asyncio
    async def test_discriminative_power_with_runner(self, length_based_grader, dataset):
        """Test the grader's ability to distinguish between high and low quality responses (using Runner)"""

        # Create a new grader instance with mapper
        length_based_grader_with_mapper = FunctionGrader(
            func=length_based_grader.func,
            name=length_based_grader.name,
            mapper={
                "query": "query",
                "response": "response",
            },
        )

        runner = GradingRunner(graders={"length_scorer": length_based_grader_with_mapper})

        # Prepare test data
        test_data = dataset
        human_scores = [item["human_score"] for item in dataset]

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset=test_data)

        # Use AccuracyAnalyzer to calculate accuracy metrics
        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=test_data,
            grader_results=results["length_scorer"],
            label_path="human_score",
        )

        # Use PrecisionAnalyzer to calculate precision metrics
        precision_analyzer = PrecisionAnalyzer()
        precision_result = precision_analyzer.analyze(
            dataset=test_data,
            grader_results=results["length_scorer"],
            label_path="human_score",
        )

        # Use RecallAnalyzer to calculate recall metrics
        recall_analyzer = RecallAnalyzer()
        recall_result = recall_analyzer.analyze(
            dataset=test_data,
            grader_results=results["length_scorer"],
            label_path="human_score",
        )

        # Use F1ScoreAnalyzer to calculate F1 score metrics
        f1_analyzer = F1ScoreAnalyzer()
        f1_result = f1_analyzer.analyze(
            dataset=test_data,
            grader_results=results["length_scorer"],
            label_path="human_score",
        )

        # Use CorrelationAnalyzer to calculate correlation metrics
        correlation_analyzer = CorrelationAnalyzer()
        correlation_result = correlation_analyzer.analyze(
            dataset=test_data,
            grader_results=results["length_scorer"],
            label_path="human_score",
        )

        # Verify analysis results contain necessary metadata
        assert "explanation" in accuracy_result.metadata
        assert "explanation" in precision_result.metadata
        assert "explanation" in recall_result.metadata
        assert "explanation" in f1_result.metadata
        assert "explanation" in correlation_result.metadata

        assert accuracy_result.name == "Accuracy Analysis"
        assert precision_result.name == "Precision Analysis"
        assert recall_result.name == "Recall Analysis"
        assert f1_result.name == "F1 Score Analysis"
        assert correlation_result.name == "Correlation Analysis"

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases like empty responses"""

        def edge_case_scorer(query: str, response: str) -> GraderScore:
            if len(response) == 0:
                return GraderScore(name="edge_case_scorer", score=0.0, reason="Empty response")
            else:
                return GraderScore(
                    name="edge_case_scorer",
                    score=min(len(response) / 10.0, 10.0),
                    reason="Non-empty response",
                )

        grader = FunctionGrader(
            func=edge_case_scorer,
            name="edge_case_scorer",
            mode=GraderMode.POINTWISE,
        )

        # Test empty response
        result = await grader.aevaluate(
            query="Test query",
            response="",
        )
        assert result.score == 0.0
        assert result.reason == "Empty response"

        # Test non-empty response
        result = await grader.aevaluate(
            query="Test query",
            response="Non-empty",
        )
        assert result.score > 0.0
        assert result.reason == "Non-empty response"
