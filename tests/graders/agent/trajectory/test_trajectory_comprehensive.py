#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for TrajectoryComprehensiveGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using TrajectoryComprehensiveGrader as an example:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real API)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_comprehensive.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_comprehensive.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_comprehensive.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock

import pytest

from openjudge.graders.agent.trajectory.trajectory_comprehensive import (
    TrajectoryComprehensiveGrader,
)
from openjudge.graders.base_grader import GraderError
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestTrajectoryComprehensiveGraderUnit:
    """Unit tests for TrajectoryComprehensiveGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()

        grader = TrajectoryComprehensiveGrader(
            model=mock_model,
            resolution_threshold=0.75,
        )
        assert grader.resolution_threshold == 0.75
        assert grader.name == "trajectory_comprehensive"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""
        # Setup mock response for step evaluations
        mock_step_response = AsyncMock()
        mock_step_response.parsed = {
            "step_index": 0,
            "step_description": "Call weather API",
            "contribution_score": 5,
            "relevance_score": 5,
            "accuracy_score": 5,
            "efficiency_score": 4,
            "step_reason": "Successfully retrieved weather data",
        }

        # Setup mock response for overall evaluation
        mock_overall_response = AsyncMock()
        mock_overall_response.parsed = {
            "score": 1,
            "reason": "Excellent problem solving with efficient tool usage",
            "is_resolved": True,
            "step_evaluations": [mock_step_response.parsed],
        }

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=[mock_overall_response])

        # Create grader
        grader = TrajectoryComprehensiveGrader(model=mock_model)

        # Execute test
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What's the weather in Beijing?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Beijing"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": "Sunny, 20°C",
            },
            {
                "role": "assistant",
                "content": "The weather in Beijing is sunny with a temperature of 20°C.",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert result is not None
        assert result.name == "trajectory_comprehensive"
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert "step_evaluations" in result.metadata

        # Verify model was called
        assert mock_model.achat.call_count >= 1

    @pytest.mark.asyncio
    async def test_empty_input_edge_case(self):
        """Test edge case with empty input"""
        mock_model = AsyncMock()
        grader = TrajectoryComprehensiveGrader(model=mock_model)

        # Execute test with empty messages
        result = await grader.aevaluate(messages=[])

        # Assertions
        assert isinstance(result, GraderError)
        assert "empty" in result.error.lower() or "empty" in str(result.metadata).lower()
        assert result.metadata.get("step_evaluations", []) == []

        # Model should not be called for empty input
        mock_model.achat.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_tool_calls_edge_case(self):
        """Test edge case with no tool calls"""
        mock_model = AsyncMock()
        grader = TrajectoryComprehensiveGrader(model=mock_model)

        # Execute test with messages without tool calls
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert isinstance(result, GraderError)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Setup mock to raise exception
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error: Connection timeout"))

        # Create grader
        grader = TrajectoryComprehensiveGrader(model=mock_model)

        # Execute test
        messages = [
            {"role": "user", "content": "Test query"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
                "content": "",
            },
            {"role": "tool", "name": "test", "content": "result"},
            {"role": "assistant", "content": "Done"},
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert isinstance(result, GraderError)
        assert "error" in result.error.lower() or "error" in str(result.metadata).lower()

    @pytest.mark.asyncio
    async def test_malformed_messages_error_handler(self):
        """Test error handling with malformed messages"""
        mock_model = AsyncMock()
        grader = TrajectoryComprehensiveGrader(model=mock_model)

        # Execute test with malformed messages
        malformed_messages = [
            {"role": "user"},  # Missing content
            {"content": "Hello"},  # Missing role
        ]

        result = await grader.aevaluate(messages=malformed_messages)

        # Assertions
        print(type(result))
        assert isinstance(result, GraderError)

    def test_trajectory_extraction_format(self):
        """Test trajectory extraction format correctness"""
        mock_model = AsyncMock()
        grader = TrajectoryComprehensiveGrader(model=mock_model)

        messages = [
            {"role": "system", "content": "System prompt here"},
            {"role": "user", "content": "What's the weather in Beijing?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Beijing"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": "Sunny, 20°C",
            },
            {
                "role": "assistant",
                "content": "The weather in Beijing is sunny with a temperature of 20°C.",
            },
        ]

        # Extract trajectory
        user_query, trajectory_messages, final_answer = grader._extract_trajectory_from_messages(
            messages,
            language=LanguageEnum.EN,
        )

        # Assertions
        assert user_query == "What's the weather in Beijing?"
        assert final_answer == "The weather in Beijing is sunny with a temperature of 20°C."
        assert "Step 0" in trajectory_messages or "step" in trajectory_messages.lower()
        assert "get_weather" in trajectory_messages
        assert "Beijing" in trajectory_messages
        assert "Sunny, 20°C" in trajectory_messages

    def test_multi_turn_trajectory_extraction(self):
        """Test multi-turn conversation trajectory extraction"""
        mock_model = AsyncMock()
        grader = TrajectoryComprehensiveGrader(model=mock_model)

        # Multi-turn messages
        multi_turn_messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First query"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "tool1", "arguments": "{}"}}],
                "content": "",
            },
            {"role": "tool", "name": "tool1", "content": "Result 1"},
            {"role": "assistant", "content": "Let me check more info."},
            {
                "role": "assistant",
                "tool_calls": [{"id": "2", "type": "function", "function": {"name": "tool2", "arguments": "{}"}}],
                "content": "",
            },
            {"role": "tool", "name": "tool2", "content": "Result 2"},
            {"role": "user", "content": "Additional question"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "3", "type": "function", "function": {"name": "tool3", "arguments": "{}"}}],
                "content": "",
            },
            {"role": "tool", "name": "tool3", "content": "Result 3"},
            {"role": "assistant", "content": "Final comprehensive answer here."},
        ]

        user_query, trajectory, final_answer = grader._extract_trajectory_from_messages(
            multi_turn_messages,
            language=LanguageEnum.EN,
        )

        # Assertions
        assert user_query == "First query"
        assert final_answer == "Final comprehensive answer here."
        assert "tool1" in trajectory or "Step 0" in trajectory
        assert "tool2" in trajectory or "Step 1" in trajectory
        assert "tool3" in trajectory or "Step 2" in trajectory
        assert "Additional question" in trajectory  # Follow-up user questions should be included


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(
    not RUN_QUALITY_TESTS,
    reason="Requires API keys and base URL to run quality tests",
)
@pytest.mark.quality
class TestTrajectoryComprehensiveGraderQuality:
    """Quality tests for TrajectoryComprehensiveGrader - testing evaluation quality"""

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {
                "model": "qwen3-32b",
                "api_key": OPENAI_API_KEY,
                "max_tokens": 4096,
            }
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # This shouldn't happen because tests are skipped if keys aren't configured
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_simple_trajectory_quality(self, model):
        """Test evaluation quality on simple trajectory"""
        # Create grader with real model
        grader = TrajectoryComprehensiveGrader(model=model)

        # Test data - simple weather query
        messages = [
            {"role": "system", "content": "You are a helpful assistant with weather tools"},
            {"role": "user", "content": "What's the weather in Beijing?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Beijing"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": '{"temperature": "20°C", "condition": "Sunny", "humidity": "45%"}',
            },
            {
                "role": "assistant",
                "content": "The weather in Beijing is sunny with a temperature of 20°C and humidity at 45%.",
            },
        ]

        # Execute evaluation with real model
        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert result.name == "trajectory_comprehensive"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

        # Verify parsed structure - this is in metadata for TrajectoryComprehensiveGrader
        assert "step_evaluations" in result.metadata
        step_evals = result.metadata["step_evaluations"]
        assert isinstance(step_evals, list)

        # For a simple successful query, expect good score
        assert result.score >= 0.6, f"Simple successful query should score >= 0.6, got {result.score}"

        print(f"\n=== Simple Trajectory Quality Test ===")
        print(f"Score: {result.score:.2f}")
        print(f"Reason: {result.reason}")
        print(f"Steps Evaluated: {len(step_evals)}")
        print(f"Is Resolved: {result.metadata.get('is_resolved')}")

    @pytest.mark.asyncio
    async def test_complex_multiturn_trajectory_quality(self, model):
        """Test evaluation quality on complex multi-turn trajectory"""
        # Create grader with real model
        grader = TrajectoryComprehensiveGrader(model=model, resolution_threshold=0.75)

        # Test data - complex investment research scenario
        messages = [
            {
                "role": "system",
                "content": "You are a professional financial analyst assistant with access to various financial data tools.",
            },
            {
                "role": "user",
                "content": "我想投资特斯拉，帮我做一个全面的投资分析，包括公司基本面、财务状况、行业竞争力和最新动态。",
            },
            # Step 0: Get company basic info
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_company_profile",
                            "arguments": '{"ticker": "TSLA", "fields": ["name", "sector", "industry", "employees", "description"]}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_company_profile",
                "content": '{"name": "Tesla Inc.", "ticker": "TSLA", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers", "employees": 127855, "description": "Tesla designs, develops, manufactures, and sells electric vehicles and energy generation and storage systems."}',
            },
            # Step 1: Get financial data
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_financial_metrics",
                            "arguments": '{"ticker": "TSLA", "metrics": ["revenue", "net_income", "eps", "pe_ratio", "debt_to_equity"]}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_financial_metrics",
                "content": '{"ticker": "TSLA", "revenue": "96.77B", "net_income": "14.95B", "eps": 4.73, "pe_ratio": 52.3, "debt_to_equity": 0.17, "period": "TTM"}',
            },
            # Step 2: Get competitors analysis
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "get_competitors",
                            "arguments": '{"ticker": "TSLA", "limit": 5}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_competitors",
                "content": '{"competitors": [{"name": "BYD", "ticker": "BYDDY", "market_share": "18.2%"}, {"name": "Volkswagen", "ticker": "VWAGY", "market_share": "12.1%"}, {"name": "BMW", "ticker": "BMWYY", "market_share": "8.5%"}, {"name": "NIO", "ticker": "NIO", "market_share": "4.2%"}]}',
            },
            # Step 3: Get latest news
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_4",
                        "type": "function",
                        "function": {
                            "name": "get_company_news",
                            "arguments": '{"ticker": "TSLA", "days": 7, "limit": 3}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_company_news",
                "content": '{"news": [{"title": "Tesla Cybertruck production ramps up to 2000 units per week", "date": "2024-12-07", "sentiment": "positive"}, {"title": "Tesla expands Supercharger network in Europe", "date": "2024-12-05", "sentiment": "positive"}, {"title": "Musk announces new affordable EV model for 2025", "date": "2024-12-04", "sentiment": "positive"}]}',
            },
            # Intermediate assistant thinking
            {
                "role": "assistant",
                "content": "我已经收集了特斯拉的基本信息、财务指标、竞争对手和最新动态。让我再查询一下股价走势和技术指标。",
            },
            # Step 4: Get stock price trend
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_5",
                        "type": "function",
                        "function": {
                            "name": "get_stock_price",
                            "arguments": '{"ticker": "TSLA", "period": "6M", "indicators": ["SMA50", "SMA200", "RSI"]}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_stock_price",
                "content": '{"current_price": 248.50, "change_percent": "+12.3%", "period": "6M", "high": 265.80, "low": 138.80, "SMA50": 235.20, "SMA200": 198.50, "RSI": 64.2}',
            },
            # User asks follow-up question
            {
                "role": "user",
                "content": "特斯拉的现金流情况怎么样？这对投资决策很重要。",
            },
            # Step 5: Get cash flow data (multi-turn scenario)
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_6",
                        "type": "function",
                        "function": {
                            "name": "get_cash_flow",
                            "arguments": '{"ticker": "TSLA", "statement_type": "operating_cash_flow", "periods": 4}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_cash_flow",
                "content": '{"operating_cash_flow": [{"quarter": "Q3 2024", "value": "6.25B"}, {"quarter": "Q2 2024", "value": "5.93B"}, {"quarter": "Q1 2024", "value": "2.51B"}, {"quarter": "Q4 2023", "value": "4.37B"}], "free_cash_flow": "4.22B", "trend": "improving"}',
            },
            # Step 6: Get analyst ratings
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_7",
                        "type": "function",
                        "function": {
                            "name": "get_analyst_ratings",
                            "arguments": '{"ticker": "TSLA"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "get_analyst_ratings",
                "content": '{"consensus": "Hold", "buy": 18, "hold": 22, "sell": 8, "average_price_target": 255.00, "upside_potential": "2.6%"}',
            },
            # Final comprehensive answer
            {
                "role": "assistant",
                "content": """基于我的全面分析，以下是特斯拉(TSLA)的投资评估：

**公司基本面（优秀）**
- 特斯拉是全球领先的电动汽车制造商，拥有12.8万员工
- 在电动车行业处于领先地位，市场份额优势明显

**财务表现（良好）**
- 营收967.7亿美元，净利润149.5亿美元，显示强劲盈利能力
- EPS 4.73，市盈率52.3倍（相对行业偏高，反映市场对增长的预期）
- 债务股权比0.17，财务结构稳健
- 经营现金流持续改善，最新季度达62.5亿美元，自由现金流42.2亿美元健康

**行业竞争力（强）**
- 面临比亚迪、大众等强劲竞争，但技术和品牌优势明显
- Cybertruck产能提升至每周2000辆，产品线扩张顺利
- 充电网络持续扩张，构建生态护城河

**股价技术面（积极）**
- 近6个月上涨12.3%，股价248.5美元突破50日均线
- RSI 64.2显示股价较强但未超买
- 分析师目标价255美元，上涨空间2.6%

**投资建议**
综合评估，特斯拉基本面扎实，现金流改善，成长性良好。但当前估值偏高（PE 52.3），适合长期看好电动车行业的投资者。建议：
1. 长期投资者可以逢低分批建仓
2. 短期投资者注意估值风险
3. 建议仓位控制在组合的5-10%
4. 关注季度交付量数据和新车型进展

风险提示：宏观经济波动、竞争加剧、估值回调风险。""",
            },
        ]

        # Execute evaluation with real model
        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert result.name == "trajectory_comprehensive"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

        # Verify step evaluations exist in metadata
        step_evals = result.metadata.get("step_evaluations", [])
        assert isinstance(step_evals, list)

        # Should have evaluated multiple steps (7 tool calls)
        assert len(step_evals) >= 5, f"Complex trajectory should have >= 5 steps, got {len(step_evals)}"

        # Verify parsed
        assert "is_resolved" in result.metadata

        # For a comprehensive research query, expect good score
        assert result.score >= 0.6, f"Comprehensive research should score >= 0.6, got {result.score}"

        # Print detailed results
        print(f"\n=== Complex Multi-turn Trajectory Quality Test ===")
        print(f"Overall Score: {result.score:.2f}")
        print(f"Number of Steps Evaluated: {len(step_evals)}")
        print(f"Resolution Status: {result.metadata.get('is_resolved')}")
        print(f"\nStep-by-Step Scores:")

        for step in step_evals[:3]:  # Print first 3 steps
            print(f"\nStep {step.get('step_index')}: {step.get('step_description', 'N/A')[:80]}")
            print(f"  Contribution: {step.get('contribution_score', 0):.2f}")
            print(f"  Relevance: {step.get('relevance_score', 0):.2f}")
            print(f"  Accuracy: {step.get('accuracy_score', 0):.2f}")
            print(f"  Efficiency: {step.get('efficiency_score', 0):.2f}")

        print(f"\nOverall Reason: {result.reason[:200]}...")

    @pytest.mark.asyncio
    async def test_consistency_quality(self, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = TrajectoryComprehensiveGrader(model=model)

        # Test data - same trajectory evaluated twice
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What's the capital of France?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "search_knowledge", "arguments": '{"query": "capital of France"}'},
                    },
                ],
                "content": "",
            },
            {"role": "tool", "name": "search_knowledge", "content": "Paris is the capital of France."},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]

        # Run evaluation twice
        result1 = await grader.aevaluate(messages=messages)
        result2 = await grader.aevaluate(messages=messages)

        # Assertions
        assert result1.name == result2.name
        assert isinstance(result1.score, (int, float))
        assert isinstance(result2.score, (int, float))

        # Scores should be reasonably consistent (within 0.3 range for LLM variance)
        score_diff = abs(result1.score - result2.score)
        assert score_diff <= 0.3, f"Consistency issue: scores differ by {score_diff}"

        print(f"\n=== Consistency Quality Test ===")
        print(f"Run 1 Score: {result1.score:.2f}")
        print(f"Run 2 Score: {result2.score:.2f}")
        print(f"Score Difference: {score_diff:.2f}")
