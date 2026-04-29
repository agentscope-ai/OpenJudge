#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ResponseCompletenessGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ResponseCompletenessGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/response/test_response_completeness.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/response/test_response_completeness.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/response/test_response_completeness.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import (
    AccuracyAnalyzer,
    FalseNegativeAnalyzer,
    FalsePositiveAnalyzer,
)
from openjudge.graders.agent.response import ResponseCompletenessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestResponseCompletenessGraderUnit:
    """Unit tests for ResponseCompletenessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ResponseCompletenessGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "response_completeness"
        assert grader.model == mock_model

        language_template = grader.get_template(LanguageEnum.ZH)
        assert len(language_template) == 1
        assert "zh" in language_template
        template = language_template["zh"]
        assert len(template) == 2
        assert len(template[1]) == 2
        assert template[1]["role"] == "user"
        assert template[1]["content"].startswith(
            "你是一名评估AI智能体回复的专家。你的任务是评估智能体的回复是否完整地解决了用户查询的所有方面。"
        )

        language_template = grader.get_default_template(LanguageEnum.EN)
        assert len(language_template) == 1
        assert "en" in language_template
        template = language_template["en"]
        assert len(template) == 2
        assert len(template[1]) == 2
        assert template[1]["role"] == "user"
        assert template[1]["content"].startswith(
            "You are an expert in evaluating AI agent responses. Your task is to evaluate whether the agent's response completely addresses all aspects of the user's query."
        )

    @pytest.mark.asyncio
    async def test_successful_evaluation_complete_response(self):
        """Test successful evaluation with a complete response"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "The response fully addresses all aspects of the query with sufficient detail",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="What's the weather in NYC and should I bring an umbrella?",
                response="It's sunny in NYC, 72°F. No umbrella needed as there's no rain forecast for today.",
            )

            assert result.score == 5.0
            assert "fully addresses" in result.reason.lower() or "complete" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_incomplete_response(self):
        """Test evaluation with an incomplete response that misses query aspects"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "The response only addresses the weather but omits the umbrella question",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="What's the weather in NYC and should I bring an umbrella?",
                response="It's sunny in NYC.",
            )

            assert result.score == 2.0
            assert "omits" in result.reason.lower() or "miss" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_chat_history_query(self):
        """Test evaluation when query is provided as chat history (list of dicts)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 4,
            "reason": "Mostly complete response covering the main aspects",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            query_history = [
                {"role": "user", "content": "What's the weather in NYC?"},
                {"role": "assistant", "content": "It's sunny, 72°F."},
                {"role": "user", "content": "Should I bring an umbrella?"},
            ]

            result = await grader.aevaluate(
                query=query_history,
                response="No umbrella needed, it will stay sunny all day.",
            )

            assert result.score == 4.0

    @pytest.mark.asyncio
    async def test_evaluation_with_context(self):
        """Test evaluation with optional context"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Complete response considering the provided context",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="How do I deploy a Python app?",
                response="You can deploy using Docker. Here are the steps: 1) Create a Dockerfile...",
                context="The user is deploying to AWS",
            )

            assert result.score == 5.0

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ResponseCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="What's the weather?",
                response="It's sunny.",
            )

            assert result.score == 0.0
            assert "Evaluation error: API Error" in result.reason

    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Test that metadata contains expected fields"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 4,
            "reason": "Mostly complete response",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Tell me about Python",
                response="Python is a programming language.",
            )

            assert result.metadata["raw_score"] == 4.0
            assert result.metadata["evaluation_type"] == "response_completeness"
            assert result.name == "response_completeness"


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestResponseCompletenessGraderQuality:
    """Quality tests for ResponseCompletenessGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset with complete and incomplete response examples"""
        return [
            # Case 1: Complete response - covers all aspects
            {
                "query": "What's the weather in NYC and should I bring an umbrella?",
                "response": "It's sunny in NYC, 72°F. No umbrella needed as there's no rain forecast for today.",
                "human_score": 5,
            },
            # Case 2: Incomplete response - only answers one part
            {
                "query": "What's the weather in NYC and should I bring an umbrella?",
                "response": "It's sunny in NYC.",
                "human_score": 2,
            },
            # Case 3: Complete response - multi-part question
            {
                "query": "Explain the differences between Python and JavaScript in terms of syntax, performance, and use cases.",
                "response": "Python uses indentation for blocks while JavaScript uses curly braces. Python is generally slower but more readable, while JavaScript is faster in browsers. Python is used for data science and ML, while JavaScript dominates web development.",
                "human_score": 5,
            },
            # Case 4: Partially complete - only addresses syntax
            {
                "query": "Explain the differences between Python and JavaScript in terms of syntax, performance, and use cases.",
                "response": "Python uses indentation while JavaScript uses curly braces for code blocks.",
                "human_score": 2,
            },
            # Case 5: Severely incomplete - barely addresses the query
            {
                "query": "What are the steps to deploy a Python app, configure CI/CD, and set up monitoring?",
                "response": "You can deploy it.",
                "human_score": 1,
            },
            # Case 6: Mostly complete - covers major aspects with minor gaps
            {
                "query": "What are the pros and cons of React vs Vue?",
                "response": "React has a larger ecosystem and more job opportunities. Vue is easier to learn and has simpler syntax. React uses JSX while Vue uses templates. Both are great for building UIs.",
                "human_score": 4,
            },
        ]

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen3-max-preview", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_discriminative_power_with_runner(self, dataset, model):
        """Test the grader's ability to distinguish between complete and incomplete responses"""
        grader = ResponseCompletenessGrader(model=model)

        grader_configs = {
            "response_completeness": GraderConfig(
                grader=grader,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        results = await runner.arun(dataset)

        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=dataset,
            grader_results=results["response_completeness"],
            label_path="human_score",
        )

        assert accuracy_result.accuracy >= 0.5, f"Accuracy below threshold: {accuracy_result.accuracy}"
        assert "explanation" in accuracy_result.metadata
        assert accuracy_result.name == "Accuracy Analysis"

        print(f"Accuracy: {accuracy_result.accuracy}")

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        grader = ResponseCompletenessGrader(model=model)

        grader_configs = {
            "response_completeness_run1": GraderConfig(
                grader=grader,
            ),
            "response_completeness_run2": GraderConfig(
                grader=grader,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        results = await runner.arun(dataset)

        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            dataset=dataset,
            grader_results=results["response_completeness_run1"],
            another_grader_results=results["response_completeness_run2"],
        )

        assert (
            consistency_result.consistency >= 0.7
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestResponseCompletenessGraderAdversarial:
    """Adversarial tests for ResponseCompletenessGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with complete and incomplete response pairs"""
        return [
            {
                "query": "What's the weather in NYC and should I bring an umbrella?",
                "complete_response": "It's sunny in NYC, 72°F. No umbrella needed as there's no rain forecast for today.",
                "incomplete_response": "It's sunny in NYC.",
                "complete_label": 5,
                "incomplete_label": 2,
            },
            {
                "query": "Explain the pros and cons of microservices architecture.",
                "complete_response": "Pros: independent deployment, technology diversity, scalability per service. Cons: operational complexity, network latency, data consistency challenges, harder debugging across services.",
                "incomplete_response": "Microservices are good for big companies.",
                "complete_label": 5,
                "incomplete_label": 1,
            },
            {
                "query": "How do I install, configure, and run a PostgreSQL database?",
                "complete_response": "Install with apt-get install postgresql. Configure by editing postgresql.conf for port and connections. Run with sudo systemctl start postgresql. Then connect with psql -U postgres.",
                "incomplete_response": "You can install PostgreSQL from their website.",
                "complete_label": 5,
                "incomplete_label": 1,
            },
        ]

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen3-max-preview", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_adversarial_response_completeness_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        grader = ResponseCompletenessGrader(model=model)

        grader_configs = {
            "response_completeness_complete": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "complete_response",
                },
            ),
            "response_completeness_incomplete": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "incomplete_response",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        results = await runner.arun(dataset)

        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["response_completeness_incomplete"],
            label_path="incomplete_label",
        )

        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["response_completeness_complete"],
            label_path="complete_label",
        )

        assert fp_result.false_positive_rate <= 0.5, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.5, f"False negative rate too high: {fn_result.false_negative_rate}"

        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"