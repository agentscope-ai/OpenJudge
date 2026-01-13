#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete test suite for LLMGrader following the GRADER_TESTING_STRATEGY.md guidelines.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using LLMGrader as an example:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against gold standard datasets)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/test_llm_grader.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/test_llm_grader.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/test_llm_grader.py -m quality
    ```
"""

import os
from typing import Callable, Dict, Tuple, Union
from unittest.mock import AsyncMock

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import AccuracyAnalyzer
from openjudge.graders.base_grader import (
    BaseGrader,
    GraderMode,
    GraderRank,
    GraderScore,
)
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderError
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestLLMGraderUnit:
    """Unit tests for LLMGrader - testing isolated functionality"""

    def test_initialization_failure_without_template(self):
        """Test initialization failure without template"""
        with pytest.raises(ValueError) as error_obj:
            LLMGrader(
                model=AsyncMock(),
                name="foo",
            )
        assert "Template must be a str, list, dict or PromptTemplate object" in str(error_obj.value)

    def test_initialization_with_string_template(self):
        """Test successful initialization with string template"""
        mock_model = AsyncMock()
        template_str = """You're a LLM query answer relevance grader, you'll received Query/Response:
    Query: {query}
    Response: {response}
    Please read query/response, if the Response answers the Query, return 1, return 0 if no.
    Return format, json.
    ```
    {{
        "score": score,
        "reason": "scoring reason",
    }}
    ```"""

        grader = LLMGrader(
            model=mock_model,
            name="test_llm_grader",
            template=template_str,
        )

        assert grader.name == "test_llm_grader"
        assert grader.model == mock_model

    def test_initialization_with_dict_template(self):
        """Test successful initialization with dict template"""
        mock_model = AsyncMock()
        template_dict = {
            "messages": {
                "en": [
                    {
                        "role": "system",
                        "content": "You're a LLM query answer relevance grader.",
                    },
                    {
                        "role": "user",
                        "content": """You'll received Query/Response:
    Query: {query}
    Response: {response}
    Please read query/response, if the Response answers the Query, return 1, return 0 if no.
    Return format, json.
    ```
    {{
        "score": score,
        "reason": "scoring reason",
    }}
    ```""",
                    },
                ],
            },
        }

        grader = LLMGrader(
            model=mock_model,
            name="test_llm_grader",
            template=template_dict,
        )

        assert grader.name == "test_llm_grader"
        assert grader.model == mock_model

    def test_get_metadata(self):
        meta = LLMGrader.get_metadata()
        assert len(meta) == 2
        assert "aevaluate" in meta
        assert "Evaluate using LLM." in meta["aevaluate"]
        assert "Performs evaluation using a large language model" in meta["aevaluate"]
        assert "prompt" in meta
        assert not meta["prompt"]

    def test_initialization_with_model_dict(self):
        """Test initialization with model configuration dict"""
        model_config = {
            "model": "qwen-max",
            "api_key": "test-key",
        }

        template_str = """You're a LLM query answer relevance grader, you'll received Query/Response:
    Query: {query}
    Response: {response}
    Please read query/response, if the Response answers the Query, return 1, return 0 if no.
    Return format, json.
    ```
    {{
        "score": score,
        "reason": "scoring reason",
    }}
    ```"""

        grader = LLMGrader(
            model=model_config,
            name="test_llm_grader",
            template=template_str,
        )

        assert grader.name == "test_llm_grader"
        assert isinstance(grader.model, OpenAIChatModel)
        # Note: We can't easily check the model config since it's private

    @pytest.mark.asyncio
    async def test_pointwise_evaluation_success(self):
        """Test successful pointwise evaluation with valid inputs"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 4.5, "reason": "Response is mostly accurate with minor issues"}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader with template that follows the specification in docs
        template = """You are an LLM query answer relevance grader. You will receive a Query and Response:
    Query: {query}
    Response: {response}
    Please analyze the query and response. If the response adequately answers the query, return 1; otherwise, return 0.
    Return your answer in JSON format:
    ```json
    {{
        "score": score,
        "reason": "explanation for scoring"
    }}
    ```"""

        grader = LLMGrader(
            model=mock_model,
            name="accuracy_grader",
            template=template,
            mode=GraderMode.POINTWISE,
        )

        # Execute test
        result = await grader.aevaluate(
            query="What is 2+2?",
            response="2+2 equals 4",
        )

        # Assertions
        # For pointwise mode, result should be GraderScore which has score attribute
        assert isinstance(result, GraderScore)
        assert result.score == 4.5
        assert "mostly accurate" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_listwise_evaluation_success(self):
        """Test successful listwise evaluation with valid inputs"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.parsed = {
            "rank": [2, 1, 3],
            "reason": "First response is most relevant, second is partially relevant, third is off-topic",
        }

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader with template that follows the specification in docs
        template = """You're a LLM query answer ranking grader, you'll received Query and multiple Responses:
    Query: {query}
    Responses:
    1. {response_1}
    2. {response_2}
    3. {response_3}
    Please read query/responses, rank the responses from most relevant (1) to least relevant (3).
    Return format, json.
    ```
    {{
        "rank": [ranked_indices],
        "reason": "ranking reason",
    }}
    ```"""

        grader = LLMGrader(
            model=mock_model,
            name="relevance_ranker",
            template=template,
            mode=GraderMode.LISTWISE,
        )

        # Execute test
        result = await grader.aevaluate(
            query="Best practices for Python programming",
            response_1="Use meaningful variable names and follow PEP8 guidelines",
            response_2="Python is a programming language created by Guido van Rossum",
            response_3="The weather today is sunny and warm",
        )

        # Assertions
        # For listwise mode, result should be GraderRank which has rank attribute
        assert isinstance(result, GraderRank)
        assert result.rank == [2, 1, 3]
        assert "most relevant" in result.reason.lower()

    def test_serialization_methods(self):
        """Test to_dict and from_config methods"""
        # Use actual model config for testing
        model_config = {
            "model": "qwen3-32b",
            "api_key": "test-key",
        }

        template_str = """You are an LLM response relevance evaluator. Analyze the following query and response:
    Query: {query}
    Response: {response}

    Determine if the response adequately addresses the query. Your evaluation should be based on:
    - Accuracy of information
    - Relevance to the query
    - Completeness of answer

    Return your assessment in JSON format with a score from 0-5 and explanation:
    ```json
    {{
        "score": score,
        "reason": "detailed explanation for scoring"
    }}
    ```"""

        original_grader = LLMGrader(
            model=model_config,
            name="serialization_test",
            template=template_str,
        )

        # Test to_dict
        config = original_grader.to_dict()
        assert config["name"] == "serialization_test"
        assert "template" in config

        # Update config to use a valid model for reconstruction
        config["model"] = {
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
        }

        # Test from_config
        reconstructed_grader = LLMGrader.from_config(config)
        assert reconstructed_grader.name == "serialization_test"


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

pytestmark = pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")


@pytest.mark.quality
class TestLLMGraderQuality:
    """Quality tests for LLMGrader - testing evaluation quality using golden dataset"""

    @pytest.fixture
    def dataset(self):
        """Load gold standard dataset"""
        return [
            {
                "query": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "context": "Geography facts",
                "human_score": 5,  # Completely accurate response
            },
            {
                "query": "What is the capital of France?",
                "response": "The capital of France is Berlin.",
                "context": "Geography facts",
                "human_score": 1,  # Incorrect response
            },
            {
                "query": "Explain quantum computing in simple terms",
                "response": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, allowing for parallel computation.",
                "context": "Physics and computing",
                "human_score": 4,  # Good but could be more comprehensive
            },
            {
                "query": "Explain quantum computing in simple terms",
                "response": "Quantum computing is a type of computing that uses quantum mechanics.",
                "context": "Physics and computing",
                "human_score": 2,  # Very vague response
            },
        ]

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen3-32b", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # This shouldn't happen because tests are skipped if keys aren't configured
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_discriminative_power_with_runner(self, dataset, model):
        """Test the grader's ability to distinguish between accurate and inaccurate responses (using Runner)"""
        # Create grader with real model following the specification in docs
        template = """You're a LLM query answer accuracy grader, you'll received Query/Response and Context:
    Query: {query}
    Response: {response}
    Context: {context}
    Please read query/response/context, rate the accuracy of the response from 1 (completely inaccurate) to 5 (completely accurate).
    Return format, json.
    ```
    {{
        "score": score,
        "reason": "scoring reason",
    }}
    ```"""

        grader = LLMGrader(
            model=model,
            name="accuracy_evaluator",
            template=template,
            mode=GraderMode.POINTWISE,
        )

        # Use mapper to configure data transformation
        grader_configs: Dict[
            str,
            Union[GraderConfig, BaseGrader, Tuple[BaseGrader, Union[Dict[str, str], Callable, None]]],
        ] = {
            "accuracy": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Prepare test data
        test_data = dataset
        # human_scores = [item["human_score"] for item in dataset]  # Not used but kept for clarity

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset=test_data)

        # Use AccuracyAnalyzer to calculate accuracy metrics
        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=test_data,
            grader_results=results["accuracy"],
            label_path="human_score",
        )

        # Assert that quality metrics meet expected thresholds
        assert accuracy_result.accuracy >= 0.6, f"Accuracy below threshold: {accuracy_result.accuracy}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in accuracy_result.metadata
        assert accuracy_result.name == "Accuracy Analysis"

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency (using Runner)"""
        # Create grader with real model following the specification in docs
        template = """You're a LLM query answer accuracy grader, you'll received Query/Response and Context:
    Query: {query}
    Response: {response}
    Context: {context}
    Please read query/response/context, rate the accuracy of the response from 1 (completely inaccurate) to 5 (completely accurate).
    Return format, json.
    ```
    {{
        "score": score,
        "reason": "scoring reason",
    }}
    ```"""

        grader = LLMGrader(
            model=model,
            name="consistency_evaluator",
            template=template,
            mode=GraderMode.POINTWISE,
        )

        # Use duplicate configuration to implement consistency testing
        grader_configs: Dict[
            str,
            Union[GraderConfig, BaseGrader, Tuple[BaseGrader, Union[Dict[str, str], Callable, None]]],
        ] = {
            "accuracy_run1": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                },
            ),
            "accuracy_run2": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Prepare test data
        test_data = dataset

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset=test_data)

        # Use ConsistencyAnalyzer to calculate consistency metrics
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            dataset=dataset,
            grader_results=results["accuracy_run1"],
            another_grader_results=results["accuracy_run2"],
        )

        # Assert that consistency metrics meet expected thresholds
        assert (
            consistency_result.consistency >= 0.8
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name is not None
        assert consistency_result.name == "Consistency Analysis"
