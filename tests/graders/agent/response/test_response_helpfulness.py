#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ResponseHelpfulnessGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ResponseHelpfulnessGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/response/test_response_helpfulness.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/response/test_response_helpfulness.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/response/test_response_helpfulness.py -m quality
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
from openjudge.graders.agent.response import ResponseHelpfulnessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestResponseHelpfulnessGraderUnit:
    """Unit tests for ResponseHelpfulnessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ResponseHelpfulnessGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "response_helpfulness"
        assert grader.model == mock_model

        language_template = grader.get_template(LanguageEnum.ZH)
        assert len(language_template) == 1
        assert "zh" in language_template
        template = language_template["zh"]
        assert len(template) == 2
        assert len(template[1]) == 2
        assert template[1]["role"] == "user"
        assert template[1]["content"].startswith(
            "你是一名评估AI智能体回复的专家。你的任务是评估智能体的回复是否有帮助且为用户提供了真正的价值。"
        )

        language_template = grader.get_default_template(LanguageEnum.EN)
        assert len(language_template) == 1
        assert "en" in language_template
        template = language_template["en"]
        assert len(template) == 2
        assert len(template[1]) == 2
        assert template[1]["role"] == "user"
        assert template[1]["content"].startswith(
            "You are an expert in evaluating AI agent responses. Your task is to evaluate whether the agent's response is helpful and provides genuine value to the user."
        )

    @pytest.mark.asyncio
    async def test_successful_evaluation_helpful_response(self):
        """Test successful evaluation with a helpful response"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Highly helpful — provides actionable, clear, and comprehensive information",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseHelpfulnessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="How do I deploy a Python app?",
                response="You can deploy using Docker. Here are the steps: 1) Create a Dockerfile, 2) Build the image with `docker build -t myapp .`, 3) Run with `docker run -p 8080:8080 myapp`. I recommend also setting up a CI/CD pipeline for automated deployments.",
            )

            assert result.score == 5.0
            assert "helpful" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_unhelpful_response(self):
        """Test evaluation with an unhelpful, minimal response"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "Not helpful — the response fails to provide useful or actionable information",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseHelpfulnessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="How do I deploy a Python app?",
                response="You can deploy it somehow.",
            )

            assert result.score == 1.0
            assert "not helpful" in result.reason.lower() or "fails to provide" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluation_with_chat_history_query(self):
        """Test evaluation when query is provided as chat history (list of dicts)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 4,
            "reason": "Helpful response with actionable information",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseHelpfulnessGrader(model=mock_model)
            grader.model.achat = mock_achat

            query_history = [
                {"role": "user", "content": "How do I set up a Python virtual environment?"},
                {"role": "assistant", "content": "Use venv module."},
                {"role": "user", "content": "Can you give me the exact commands?"},
            ]

            result = await grader.aevaluate(
                query=query_history,
                response="Run: python -m venv myenv, then source myenv/bin/activate on Mac/Linux or myenv\\Scripts\\activate on Windows.",
            )

            assert result.score == 4.0

    @pytest.mark.asyncio
    async def test_evaluation_with_context(self):
        """Test evaluation with optional context"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Highly helpful response considering the deployment context",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseHelpfulnessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="How do I deploy a Python app?",
                response="For AWS deployment, use Elastic Beanstalk or ECS. Here are the steps...",
                context="Deploying to AWS with Docker",
            )

            assert result.score == 5.0

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ResponseHelpfulnessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="How do I deploy a Python app?",
                response="You can deploy using Docker.",
            )

            assert result.score == 0.0
            assert "Evaluation error: API Error" in result.reason

    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Test that metadata contains expected fields"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 3,
            "reason": "Moderately helpful response",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ResponseHelpfulnessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="How do I learn Python?",
                response="You can read some books about Python.",
            )

            assert result.metadata["raw_score"] == 3.0
            assert result.metadata["evaluation_type"] == "response_helpfulness"
            assert result.name == "response_helpfulness"


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestResponseHelpfulnessGraderQuality:
    """Quality tests for ResponseHelpfulnessGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset with helpful and unhelpful response examples"""
        return [
            # Case 1: Highly helpful - actionable with context
            {
                "query": "How do I deploy a Python app?",
                "response": "You can deploy using Docker. Here are the steps: 1) Create a Dockerfile, 2) Build the image with `docker build -t myapp .`, 3) Run with `docker run -p 8080:8080 myapp`. I recommend also setting up a CI/CD pipeline for automated deployments.",
                "human_score": 5,
            },
            # Case 2: Not helpful - vague and non-actionable
            {
                "query": "How do I deploy a Python app?",
                "response": "You can deploy it somehow.",
                "human_score": 1,
            },
            # Case 3: Moderately helpful - correct but minimal
            {
                "query": "What's the best way to handle errors in Python?",
                "response": "Use try-except blocks to catch exceptions.",
                "human_score": 3,
            },
            # Case 4: Helpful - provides context and alternatives
            {
                "query": "What's the best way to handle errors in Python?",
                "response": "Use try-except blocks for specific exceptions. You can also use finally for cleanup code. For custom errors, create your own Exception subclass. Consider using context managers for resource cleanup as well.",
                "human_score": 4,
            },
            # Case 5: Slightly helpful - addresses query but lacks depth
            {
                "query": "How can I improve my website's performance?",
                "response": "You should make it faster.",
                "human_score": 1,
            },
            # Case 6: Highly helpful - comprehensive with actionable steps
            {
                "query": "How can I improve my website's performance?",
                "response": "1) Optimize images using WebP format and lazy loading. 2) Minify CSS and JS files. 3) Enable gzip compression. 4) Use a CDN for static assets. 5) Implement browser caching with proper Cache-Control headers. 6) Consider code splitting for large JS bundles.",
                "human_score": 5,
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
        """Test the grader's ability to distinguish between helpful and unhelpful responses"""
        grader = ResponseHelpfulnessGrader(model=model)

        grader_configs = {
            "response_helpfulness": GraderConfig(
                grader=grader,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        results = await runner.arun(dataset)

        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=dataset,
            grader_results=results["response_helpfulness"],
            label_path="human_score",
        )

        assert accuracy_result.accuracy >= 0.5, f"Accuracy below threshold: {accuracy_result.accuracy}"
        assert "explanation" in accuracy_result.metadata
        assert accuracy_result.name == "Accuracy Analysis"

        print(f"Accuracy: {accuracy_result.accuracy}")

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        grader = ResponseHelpfulnessGrader(model=model)

        grader_configs = {
            "response_helpfulness_run1": GraderConfig(
                grader=grader,
            ),
            "response_helpfulness_run2": GraderConfig(
                grader=grader,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        results = await runner.arun(dataset)

        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            dataset=dataset,
            grader_results=results["response_helpfulness_run1"],
            another_grader_results=results["response_helpfulness_run2"],
        )

        assert (
            consistency_result.consistency >= 0.7
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestResponseHelpfulnessGraderAdversarial:
    """Adversarial tests for ResponseHelpfulnessGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with helpful and unhelpful response pairs"""
        return [
            {
                "query": "How do I deploy a Python app?",
                "helpful_response": "You can deploy using Docker. Here are the steps: 1) Create a Dockerfile, 2) Build the image, 3) Run the container. For production, consider using AWS ECS or GCP Cloud Run.",
                "unhelpful_response": "You can deploy it somehow.",
                "helpful_label": 5,
                "unhelpful_label": 1,
            },
            {
                "query": "What's the best way to handle errors in Python?",
                "helpful_response": "Use try-except blocks for specific exceptions, finally for cleanup, and create custom Exception subclasses when needed. Consider logging errors for debugging.",
                "unhelpful_response": "Just catch the errors.",
                "helpful_label": 4,
                "unhelpful_label": 1,
            },
            {
                "query": "How can I optimize my SQL queries?",
                "helpful_response": "1) Add proper indexes on frequently queried columns. 2) Use EXPLAIN to analyze query plans. 3) Avoid SELECT *. 4) Use JOINs instead of subqueries when possible. 5) Consider query caching.",
                "unhelpful_response": "Make them faster.",
                "helpful_label": 5,
                "unhelpful_label": 1,
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
    async def test_adversarial_response_helpfulness_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        grader = ResponseHelpfulnessGrader(model=model)

        grader_configs = {
            "response_helpfulness_helpful": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "helpful_response",
                },
            ),
            "response_helpfulness_unhelpful": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "unhelpful_response",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        results = await runner.arun(dataset)

        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["response_helpfulness_unhelpful"],
            label_path="unhelpful_label",
        )

        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["response_helpfulness_helpful"],
            label_path="helpful_label",
        )

        assert fp_result.false_positive_rate <= 0.5, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.5, f"False negative rate too high: {fn_result.false_negative_rate}"

        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"