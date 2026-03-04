# -*- coding: utf-8 -*-
"""
Complete demo test for SearchCorrectnessGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using SearchCorrectnessGrader as an example of AgenticGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/common/test_search_correctness.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/common/test_search_correctness.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/common/test_search_correctness.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from openjudge.graders.common.search_correctness import SearchCorrectnessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestSearchCorrectnessGraderUnit:
    """Unit tests for SearchCorrectnessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = SearchCorrectnessGrader(
            model=mock_model,
            tavily_api_key="test-api-key",
        )
        assert grader.name == "search_correctness"
        # model is now inside agent (unified interface design)
        assert grader.agent.model == mock_model

    def test_initialization_with_language(self):
        """Test initialization with different languages"""
        mock_model = AsyncMock()

        # Test English
        grader_en = SearchCorrectnessGrader(
            model=mock_model,
            tavily_api_key="test-api-key",
            language=LanguageEnum.EN,
        )
        assert grader_en.language == LanguageEnum.EN

        # Test Chinese
        grader_zh = SearchCorrectnessGrader(
            model=mock_model,
            tavily_api_key="test-api-key",
            language=LanguageEnum.ZH,
        )
        assert grader_zh.language == LanguageEnum.ZH

    @pytest.mark.asyncio
    async def test_search_correctness_grader_with_mock(self):
        """Test SearchCorrectnessGrader with mocked model and search tool"""
        # Setup mock model
        mock_model = MagicMock()

        # Mock the chat response (no tool calls, direct response)
        # AgenticGrader's _react_loop accesses response.content and response.tool_calls directly
        mock_response = MagicMock()
        mock_response.content = '{"score": 4, "reason": "The response is mostly accurate based on search results."}'
        mock_response.tool_calls = None  # No tool calls means final answer

        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = SearchCorrectnessGrader(
            model=mock_model,
            tavily_api_key="test-api-key",
            language=LanguageEnum.EN,
        )

        # Test evaluation
        result = await grader.aevaluate(
            query="What is the capital of France?",
            response="Paris is the capital of France.",
        )

        # Verify result structure
        assert result.name == "search_correctness"
        assert isinstance(result.score, (int, float))
        assert result.score >= 1 and result.score <= 5

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        mock_model = MagicMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error"))

        grader = SearchCorrectnessGrader(
            model=mock_model,
            tavily_api_key="test-api-key",
        )

        result = await grader.aevaluate(
            query="What is the capital of France?",
            response="Paris is the capital of France.",
        )

        # Assertions - error is returned as GraderError with error field
        assert result.error is not None
        assert "API Error" in result.error


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(TAVILY_API_KEY and OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires TAVILY_API_KEY and OpenAI API keys to run quality tests")
@pytest.mark.quality
class TestSearchCorrectnessGraderQuality:
    """Quality tests for SearchCorrectnessGrader - testing evaluation quality"""

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen3-32b", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_accurate_information(self, model):
        """Test: Grader correctly evaluates accurate information"""
        grader = SearchCorrectnessGrader(
            model=model,
            tavily_api_key=TAVILY_API_KEY,
            language=LanguageEnum.ZH,
        )

        query = "介绍一下紫金矿业2024年的经营情况和主要业绩"
        response = """紫金矿业2024年经营情况如下：
1. 营业收入：全年实现营业收入约3200亿元，同比增长约15%
2. 净利润：归属于上市公司股东的净利润约320亿元，创历史新高
3. 主要产品产量：
   - 矿产金产量约75吨，同比增长约10%
   - 矿产铜产量约115万吨，同比增长约8%
   - 矿产锌产量约48万吨
4. 重大项目进展：
   - 刚果(金)卡莫阿-卡库拉铜矿三期建设顺利推进
   - 西藏巨龙铜矿一期投产
   - 塞尔维亚丘卡卢-佩吉铜金矿产能持续提升
5. 市值表现：公司市值突破4000亿元，位居全球矿业公司前列"""

        result = await grader.aevaluate(query=query, response=response)

        # Verify result structure
        assert result.name == "search_correctness"
        assert isinstance(result.score, (int, float))
        assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
        assert len(result.reason) > 0, "Reason should not be empty"

        # Print for debugging
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"Score: {result.score}")
        print(f"Reason: {result.reason}")
        print(f"Tool Calls: {result.metadata.get('tool_calls', 0)}")
        print(f"{'=' * 60}\n")

    @pytest.mark.asyncio
    async def test_inaccurate_information(self, model):
        """Test: Grader correctly identifies inaccurate information"""
        grader = SearchCorrectnessGrader(
            model=model,
            tavily_api_key=TAVILY_API_KEY,
            language=LanguageEnum.EN,
        )

        query = "What is the population of Tokyo?"
        # Intentionally incorrect response
        response = "Tokyo has a population of about 500,000 people, making it a small city."

        result = await grader.aevaluate(query=query, response=response)

        # Verify result structure
        assert result.name == "search_correctness"
        assert isinstance(result.score, (int, float))
        assert 1 <= result.score <= 5, f"Score out of range: {result.score}"
        assert len(result.reason) > 0, "Reason should not be empty"

        # Inaccurate information should get a lower score
        # Tokyo's actual population is ~14 million
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"Score: {result.score}")
        print(f"Reason: {result.reason}")
        print(f"{'=' * 60}\n")

        # We expect a low score for clearly inaccurate information
        assert result.score <= 3, f"Expected low score for inaccurate info, got {result.score}"

    @pytest.mark.asyncio
    async def test_tool_usage(self, model):
        """Test: Grader uses search tool during evaluation"""
        grader = SearchCorrectnessGrader(
            model=model,
            tavily_api_key=TAVILY_API_KEY,
            language=LanguageEnum.EN,
        )

        query = "What are the latest developments in AI in 2024?"
        response = "In 2024, major AI developments include advances in large language models and multimodal AI systems."

        result = await grader.aevaluate(query=query, response=response)

        # Verify result structure
        assert result.name == "search_correctness"
        assert isinstance(result.score, (int, float))
        assert 1 <= result.score <= 5

        # Check that tool was used (for real-time information verification)
        tool_calls = result.metadata.get("tool_calls", 0)
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"Score: {result.score}")
        print(f"Tool Calls: {tool_calls}")
        print(f"{'=' * 60}\n")

        # For real-time information, we expect at least one tool call
        assert tool_calls >= 1, "Expected at least one tool call for real-time information verification"
