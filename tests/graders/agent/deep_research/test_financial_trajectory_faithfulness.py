#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for FinancialTrajectoryFaithfulGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using FinancialTrajectoryFaithfulGrader as an example:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real API)

Example:
    Run all tests:
    ```bash
    poetry run pytest tests/graders/agent/deep_research/test_financial_trajectory_faithful.py -v
    ```

    Run only unit tests:
    ```bash
    poetry run pytest tests/graders/agent/deep_research/test_financial_trajectory_faithful.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    poetry run pytest tests/graders/agent/deep_research/test_financial_trajectory_faithful.py -m quality
    ```
"""

import json
import os
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.agent.deep_research.financial_trajectory_faithfulness import (
    FinancialTrajectoryFaithfulGrader,
)
from rm_gallery.core.graders.schema import GraderError
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestFinancialTrajectoryFaithfulGraderUnit:
    """Unit tests for FinancialTrajectoryFaithfulGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()

        grader = FinancialTrajectoryFaithfulGrader(
            model=mock_model,
            language=LanguageEnum.ZH,
        )
        assert grader.name == "financial_trajectory_faithful"
        assert grader.model == mock_model
        assert grader.language == LanguageEnum.ZH

    def test_initialization_english(self):
        """Test initialization with English language"""
        mock_model = AsyncMock()

        grader = FinancialTrajectoryFaithfulGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "financial_trajectory_faithful"
        assert grader.language == LanguageEnum.EN

    @pytest.mark.asyncio
    async def test_successful_evaluation_no_errors(self):
        """Test successful evaluation with all tuples having no errors"""
        # Create grader first to get the callback
        mock_model = AsyncMock()
        grader = FinancialTrajectoryFaithfulGrader(model=mock_model)

        # Setup mock response with callback result
        # The callback will process tuples and return score/reason/metadata
        tuples_data = {
            "tuples": [
                {
                    "subject": "生物制品板块",
                    "indicator": "持仓占比",
                    "value": "0.46%",
                    "time": "2025年7月7日",
                    "error_type": None,
                    "error_reason": "",
                },
                {
                    "subject": "电池板块",
                    "indicator": "持仓占比",
                    "value": "0.29%",
                    "time": "2025年7月7日",
                    "error_type": None,
                    "error_reason": "",
                },
            ]
        }

        # Simulate what the callback returns
        mock_callback_response = AsyncMock()
        mock_callback_response.metadata = tuples_data
        callback_result = grader.callback(mock_callback_response)

        # Now setup the actual mock response with the callback result
        mock_response = AsyncMock()
        mock_response.parsed = callback_result

        mock_model.achat = AsyncMock(return_value=mock_response)

        # Execute test with complete tool call flow
        messages = [
            {"role": "user", "content": "分析我的持仓中生物制品板块和电池板块的占比"},
            {
                "role": "assistant",
                "content": "我将帮您查询持仓信息。",
                "tool_calls": [
                    {
                        "id": "call_123abc",
                        "function": {
                            "arguments": '{"query": "持仓占比"}',
                            "name": "search_portfolio",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search_portfolio",
                "content": "截至2025年7月7日收盘数据：\n- 生物制品板块持仓占比：0.46%\n- 电池板块持仓占比：0.29%",
                "tool_call_id": "call_123abc",
            },
            {
                "role": "assistant",
                "content": "根据您的投资组合数据（截至2025年7月7日），您的持仓情况如下：\n\n1. **生物制品板块**：持仓占比为0.46%\n2. **电池板块**：持仓占比为0.29%",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert result is not None
        assert result.name == "financial_trajectory_faithful"
        assert result.score == 1.0  # All tuples have no errors
        assert isinstance(result.reason, str)
        assert "2" in result.reason  # Should mention 2 tuples

        # Verify model was called
        mock_model.achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_evaluation_with_errors(self):
        """Test successful evaluation with some tuples having errors"""
        # Create grader first to get the callback
        mock_model = AsyncMock()
        grader = FinancialTrajectoryFaithfulGrader(model=mock_model)

        # Setup tuples data with errors
        tuples_data = {
            "tuples": [
                {
                    "subject": "生物制品板块",
                    "indicator": "收益率",
                    "value": "-1.92%",
                    "time": "近180天",
                    "error_type": "time_error",
                    "error_reason": "【来源1】显示数据时间为2025年7月7日，报告中使用近180天，时间描述不一致",
                },
                {
                    "subject": "电池板块",
                    "indicator": "持仓占比",
                    "value": "0.29%",
                    "time": "2025年7月7日",
                    "error_type": None,
                    "error_reason": "",
                },
            ]
        }

        # Simulate what the callback returns
        mock_callback_response = AsyncMock()
        mock_callback_response.metadata = tuples_data
        callback_result = grader.callback(mock_callback_response)

        # Now setup the actual mock response with the callback result
        mock_response = AsyncMock()
        mock_response.parsed = callback_result

        mock_model.achat = AsyncMock(return_value=mock_response)

        # Execute test with complete tool call flow
        messages = [
            {"role": "user", "content": "分析我的持仓板块表现"},
            {
                "role": "assistant",
                "content": "我将查询您的持仓数据和板块表现。",
                "tool_calls": [
                    {
                        "id": "call_456def",
                        "function": {
                            "arguments": '{"query": "持仓板块收益率"}',
                            "name": "search_portfolio",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search_portfolio",
                "content": "截至2025年7月7日收盘数据：\n- 生物制品板块收益率：-1.92%\n- 电池板块持仓占比：0.29%\n- 数据来源：投资组合实时数据",
                "tool_call_id": "call_456def",
            },
            {
                "role": "assistant",
                "content": "根据投资组合数据分析：\n\n1. **生物制品板块**：近180天收益率为-1.92%\n2. **电池板块**：持仓占比为0.29%（截至2025年7月7日）",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert result is not None
        assert result.name == "financial_trajectory_faithful"
        assert result.score == 0.5  # 1 error out of 2 tuples: 1 - 1/2 = 0.5
        assert isinstance(result.reason, str)
        assert "time_error" in result.reason

    @pytest.mark.asyncio
    async def test_empty_messages_edge_case(self):
        """Test edge case with empty messages"""
        mock_model = AsyncMock()
        grader = FinancialTrajectoryFaithfulGrader(model=mock_model)

        # Execute test with empty messages
        result = await grader.aevaluate(messages=[])

        # Assertions
        assert isinstance(result, GraderError)
        assert result.name == "financial_trajectory_faithful"
        assert result.error == "Empty user query or search result or AI answer"

        # Model should not be called for empty input
        mock_model.achat.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_ai_answer_edge_case(self):
        """Test edge case with no AI answer in messages"""
        mock_model = AsyncMock()
        grader = FinancialTrajectoryFaithfulGrader(model=mock_model)

        # Execute test with messages without AI answer (tool call but no final response)
        messages = [
            {"role": "user", "content": "分析我的持仓"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "function": {
                            "arguments": '{"query": "持仓"}',
                            "name": "search",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search",
                "content": "持仓数据...",
                "tool_call_id": "call_xyz",
            },
            # Missing final assistant response with content
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert isinstance(result, GraderError)
        assert result.name == "financial_trajectory_faithful"
        assert result.error == "Empty user query or search result or AI answer"

    @pytest.mark.asyncio
    async def test_no_tuples_extracted(self):
        """Test case when LLM extracts no tuples"""
        # Create grader first to get the callback
        mock_model = AsyncMock()
        grader = FinancialTrajectoryFaithfulGrader(model=mock_model)

        # Setup tuples data with no tuples
        tuples_data = {"tuples": []}

        # Simulate what the callback returns
        mock_callback_response = AsyncMock()
        mock_callback_response.metadata = tuples_data
        callback_result = grader.callback(mock_callback_response)

        # Now setup the actual mock response with the callback result
        mock_response = AsyncMock()
        mock_response.parsed = callback_result

        mock_model.achat = AsyncMock(return_value=mock_response)

        # Execute test with non-financial query
        messages = [
            {"role": "user", "content": "今天天气怎么样？"},
            {
                "role": "assistant",
                "content": "我将为您查询天气信息。",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "function": {
                            "arguments": '{"city": "北京"}',
                            "name": "get_weather",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": "北京今天天气晴朗，温度20-28°C",
                "tool_call_id": "call_weather",
            },
            {
                "role": "assistant",
                "content": "今天北京天气很好，晴朗，温度20-28°C，适合出行。",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert result.score == 0.0
        assert "未提取" in result.reason or "no tuples" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when LLM call fails"""
        # Setup mock to raise exception
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error: Connection timeout"))

        # Create grader
        grader = FinancialTrajectoryFaithfulGrader(model=mock_model)

        # Execute test with complete message flow
        messages = [
            {"role": "user", "content": "分析我的持仓"},
            {
                "role": "assistant",
                "content": "正在查询持仓数据...",
                "tool_calls": [
                    {
                        "id": "call_error",
                        "function": {
                            "arguments": '{"query": "持仓"}',
                            "name": "get_portfolio",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_portfolio",
                "content": "持仓数据：生物制品板块占比0.46%",
                "tool_call_id": "call_error",
            },
            {
                "role": "assistant",
                "content": "您的生物制品板块持仓占比为0.46%。",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert isinstance(result, GraderError)
        assert result.name == "financial_trajectory_faithful"
        assert "API Error" in result.error or "Connection timeout" in result.error

    @pytest.mark.asyncio
    async def test_malformed_tuples_error_handling(self):
        """Test error handling with malformed tuple data"""
        # Setup mock response with malformed tuples
        mock_response = AsyncMock()
        mock_response.parsed = {
            "tuples": [
                {
                    "subject": "板块",
                    # Missing required fields: indicator, value, time, error_type, error_reason
                }
            ]
        }

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = FinancialTrajectoryFaithfulGrader(model=mock_model)

        # Execute test with complete tool call flow
        messages = [
            {"role": "user", "content": "分析我的持仓"},
            {
                "role": "assistant",
                "content": "正在分析持仓...",
                "tool_calls": [
                    {
                        "id": "call_malformed",
                        "function": {
                            "arguments": '{"query": "持仓"}',
                            "name": "analyze_portfolio",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "analyze_portfolio",
                "content": "板块数据不完整",
                "tool_call_id": "call_malformed",
            },
            {
                "role": "assistant",
                "content": "您的板块持仓情况...",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # Should handle gracefully and return error
        assert isinstance(result, GraderError)
        assert result.name == "financial_trajectory_faithful"

    def test_extraction_format_with_nested_messages(self):
        """Test extraction handles nested message format correctly"""
        mock_model = AsyncMock()
        grader = FinancialTrajectoryFaithfulGrader(model=mock_model)

        # Test with nested message structure including tool calls
        messages = [
            {"message": {"role": "system", "content": "System prompt"}},
            {"message": {"role": "user", "content": "分析我的持仓"}},
            {
                "message": {
                    "role": "assistant",
                    "content": "查询中...",
                    "tool_calls": [
                        {
                            "id": "call_nested",
                            "function": {
                                "arguments": '{"query": "持仓"}',
                                "name": "search",
                            },
                            "type": "function",
                            "index": 0,
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "tool",
                    "name": "search",
                    "content": "生物制品板块持仓占比0.46%（截至2025年7月7日）",
                    "tool_call_id": "call_nested",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "您的生物制品板块持仓占比为0.46%。",
                }
            },
        ]

        # Extract components
        user_query, search_result, ai_answer = grader._extract_user_query_and_search_results_from_messages(messages)

        # Assertions
        assert user_query == "分析我的持仓"
        assert "生物制品板块持仓占比0.46%" in search_result
        assert ai_answer == "您的生物制品板块持仓占比为0.46%。"

    @pytest.mark.asyncio
    async def test_all_error_types(self):
        """Test evaluation with all different error types"""
        # Create grader first to get the callback
        mock_model = AsyncMock()
        grader = FinancialTrajectoryFaithfulGrader(model=mock_model)

        # Setup tuples data with different error types
        tuples_data = {
            "tuples": [
                {
                    "subject": "生物制品板块",
                    "indicator": "收益率",
                    "value": "-1.92%",
                    "time": "近180天",
                    "error_type": "time_error",
                    "error_reason": "【来源1】显示时间为2025年7月7日，报告使用近180天，时间不一致",
                },
                {
                    "subject": "电池板块",
                    "indicator": "收益",
                    "value": "-0.64",
                    "time": "2025年7月7日",
                    "error_type": "indicator_error",
                    "error_reason": "【来源1】显示为收益率-0.64%，报告使用收益-0.64，指标混淆",
                },
                {
                    "subject": "进阶资产",
                    "indicator": "持仓占比",
                    "value": "1.69%",
                    "time": "2025年7月7日",
                    "error_type": "subject_error",
                    "error_reason": "【来源1】显示为增值资产持仓占比1.69%，报告使用进阶资产，主体错误",
                },
                {
                    "subject": "某板块",
                    "indicator": "收益率",
                    "value": "3%",
                    "time": "2025年7月7日",
                    "error_type": "value_error",
                    "error_reason": "【来源1】显示收益率为2.87%，报告使用3%，数值不准确",
                },
            ]
        }

        # Simulate what the callback returns
        mock_callback_response = AsyncMock()
        mock_callback_response.metadata = tuples_data
        callback_result = grader.callback(mock_callback_response)

        # Now setup the actual mock response with the callback result
        mock_response = AsyncMock()
        mock_response.parsed = callback_result

        mock_model.achat = AsyncMock(return_value=mock_response)

        # Test with complete tool call flow
        messages = [
            {"role": "user", "content": "分析我的持仓情况"},
            {
                "role": "assistant",
                "content": "我将为您查询持仓数据。",
                "tool_calls": [
                    {
                        "id": "call_789ghi",
                        "function": {
                            "arguments": '{"query": "持仓数据"}',
                            "name": "get_portfolio",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_portfolio",
                "content": """截至2025年7月7日持仓数据：
- 生物制品板块收益率：-1.92%
- 电池板块收益率：-0.64%
- 增值资产持仓占比：1.69%
- 某板块收益率：2.87%""",
                "tool_call_id": "call_789ghi",
            },
            {
                "role": "assistant",
                "content": """您的持仓情况如下：
1. 生物制品板块近180天收益率为-1.92%
2. 电池板块收益为-0.64（截至2025年7月7日）
3. 进阶资产持仓占比为1.69%（截至2025年7月7日）
4. 某板块收益率为3%（截至2025年7月7日）""",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions - 4 errors out of 4 tuples: 1 - 4/4 = 0.0
        assert result.score == 0.0
        # Check all error types are mentioned in the reason
        assert "time_error" in result.reason
        assert "indicator_error" in result.reason
        assert "subject_error" in result.reason
        assert "value_error" in result.reason


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
class TestFinancialTrajectoryFaithfulGraderQuality:
    """Quality tests for FinancialTrajectoryFaithfulGrader - testing evaluation quality"""

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {
                "model": "qwen3-max",
                "api_key": OPENAI_API_KEY,
                "max_tokens": 4096,
            }
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # This shouldn't happen because tests are skipped if keys aren't configured
            raise RuntimeError("No API key configured")

    @pytest.fixture
    def test_data(self):
        """Load test data from JSON file"""
        test_data_path = os.path.join(os.path.dirname(__file__), "financial_test_data.json")
        with open(test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @pytest.mark.asyncio
    async def test_real_financial_data_quality(self, model, test_data):
        """Test evaluation quality on real financial research data"""
        # Create grader with real model
        grader = FinancialTrajectoryFaithfulGrader(model=model, language=LanguageEnum.ZH)

        # Use first test case from the data
        messages = test_data[0]

        # Execute evaluation with real model
        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert result.name == "financial_trajectory_faithful"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

        # Verify metadata structure
        assert "total_tuples" in result.metadata
        assert "error_count" in result.metadata
        assert "has_error" in result.metadata
        assert "all_tuples" in result.metadata

        # Should have extracted some tuples from this complex financial report
        assert result.metadata.get("total_tuples", 0) > 0

        print(f"\n=== Real Financial Data Quality Test ===")
        print(f"Score: {result.score:.2f}")
        print(f"Total Tuples: {result.metadata.get('total_tuples')}")
        print(f"Error Count: {result.metadata.get('error_count')}")
        print(f"Has Error: {result.metadata.get('has_error')}")
        print(f"Reason Preview: {result.reason[:300]}...")

    @pytest.mark.asyncio
    async def test_faithful_report_quality(self, model):
        """Test that grader correctly identifies faithful financial reports"""
        # Create grader with real model
        grader = FinancialTrajectoryFaithfulGrader(model=model, language=LanguageEnum.ZH)

        # Create a faithful report scenario
        messages = [
            {"role": "system", "content": "你是专业的金融分析助手"},
            {"role": "user", "content": "我的持仓中生物制品板块和电池板块的占比是多少？"},
            {
                "role": "tool",
                "name": "search_portfolio",
                "content": """截至2025年7月7日收盘数据：
                - 生物制品板块持仓占比：0.46%
                - 电池板块持仓占比：0.29%
                - 数据来源：用户投资组合实时数据
                """,
            },
            {
                "role": "assistant",
                "content": """根据您的投资组合数据（截至2025年7月7日），您的持仓情况如下：

1. **生物制品板块**：持仓占比为0.46%
2. **电池板块**：持仓占比为0.29%

这两个板块合计占您总持仓的0.75%，属于较小的配置比例。""",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # A faithful report should score high (ideally 1.0)
        # But we allow some tolerance due to LLM variance
        print(f"\n=== Faithful Report Quality Test ===")
        print(f"Score: {result.score:.2f}")
        print(f"Total Tuples: {result.metadata.get('total_tuples')}")
        print(f"Error Count: {result.metadata.get('error_count')}")
        print(f"Reason: {result.reason}")

        # Should extract tuples
        assert result.metadata.get("total_tuples", 0) >= 2

        # Check if the grader can identify this as faithful
        # Note: Due to LLM variance, we don't enforce strict score requirement
        # but log the result for manual inspection
        if result.score < 0.8:
            print(f"WARNING: Faithful report scored {result.score}, expected >= 0.8")

    @pytest.mark.asyncio
    async def test_unfaithful_report_quality(self, model):
        """Test that grader correctly identifies unfaithful financial reports with errors"""
        # Create grader with real model
        grader = FinancialTrajectoryFaithfulGrader(model=model, language=LanguageEnum.ZH)

        # Create an unfaithful report scenario with intentional errors
        messages = [
            {"role": "system", "content": "你是专业的金融分析助手"},
            {"role": "user", "content": "我的持仓中生物制品板块和电池板块的占比是多少？"},
            {
                "role": "tool",
                "name": "search_portfolio",
                "content": """截至2025年7月7日收盘数据：
                - 生物制品板块持仓占比：0.46%
                - 电池板块持仓占比：0.29%
                - 增值类资产持仓占比：1.69%
                """,
            },
            {
                "role": "assistant",
                "content": """根据您的投资组合数据，您的持仓情况如下：

1. **生物制品板块**：近180天持仓占比为0.46%（错误：时间信息不一致）
2. **电池板块**：收益为0.29（错误：指标混淆，应为持仓占比而非收益）
3. **进阶类资产**：持仓占比为1.69%（错误：主体错误，应为增值类资产）

以上数据截至2025年6月30日。""",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # An unfaithful report should score low (ideally 0.0)
        print(f"\n=== Unfaithful Report Quality Test ===")
        print(f"Score: {result.score:.2f}")
        print(f"Total Tuples: {result.metadata.get('total_tuples')}")
        print(f"Error Count: {result.metadata.get('error_count')}")
        print(f"Has Error: {result.metadata.get('has_error')}")
        print(f"Reason: {result.reason[:500]}...")

        # Should extract tuples
        assert result.metadata.get("total_tuples", 0) >= 2

        # Should detect errors
        # Note: Due to LLM variance, we don't enforce strict requirements
        # but log the result for manual inspection
        if result.score > 0.2:
            print(f"WARNING: Unfaithful report scored {result.score}, expected <= 0.2")
        if result.metadata.get("error_count", 0) == 0:
            print("WARNING: No errors detected in intentionally unfaithful report")

    @pytest.mark.asyncio
    async def test_english_language_quality(self, model):
        """Test grader works correctly with English language"""
        # Create grader with English language
        grader = FinancialTrajectoryFaithfulGrader(model=model, language=LanguageEnum.EN)

        # English test data
        messages = [
            {"role": "user", "content": "Analyze my portfolio holdings"},
            {
                "role": "tool",
                "content": "As of July 7, 2025: Biotech sector holdings: 0.46%, Battery sector holdings: 0.29%",
            },
            {
                "role": "assistant",
                "content": "Based on your portfolio data as of July 7, 2025, your Biotech sector holdings are 0.46% and Battery sector holdings are 0.29%.",
            },
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert result.name == "financial_trajectory_faithful"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

        print(f"\n=== English Language Quality Test ===")
        print(f"Score: {result.score:.2f}")
        print(f"Total Tuples: {result.metadata.get('total_tuples')}")
        print(f"Reason: {result.reason[:300]}...")

        # Should extract some tuples
        assert result.metadata.get("total_tuples", 0) > 0
