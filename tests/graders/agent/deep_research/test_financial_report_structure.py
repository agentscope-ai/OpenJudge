"""
Unit and Quality Tests for Financial Report Structure and Readability Grader

This module provides comprehensive tests for FinancialReportStructureGrader,
including unit tests with mocked dependencies and quality tests with real API calls.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.agent.deep_research.financial_report_structure import (
    FinancialReportStructureGrader,
)
from rm_gallery.core.graders.schema import GraderError, GraderScore
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

# Check for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.unit
class TestFinancialReportStructureGraderUnit:
    """Unit tests for FinancialReportStructureGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization of the grader"""
        mock_model = AsyncMock()
        grader = FinancialReportStructureGrader(model=mock_model)

        assert grader.name == "financial_report_structure"
        assert grader.description == "Financial deep research report structure and readability evaluation"

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 5, "reason": "报告结构清晰，层次分明，语言专业易懂"}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = FinancialReportStructureGrader(model=mock_model)

        # Example messages with tool calls
        messages = [
            {"role": "user", "content": "分析贵州茅台的财务状况"},
            {
                "role": "assistant",
                "content": "我将从多个维度分析贵州茅台的财务状况。",
                "tool_calls": [
                    {
                        "id": "call_comprehensive_analysis",
                        "function": {
                            "arguments": '{"symbol": "600519", "dimensions": ["营收", "盈利"]}',
                            "name": "comprehensive_analysis",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "comprehensive_analysis",
                "content": "2024年财报数据：营收1583.09亿元，净利润862.28亿元，ROE 36%",
                "tool_call_id": "call_comprehensive_analysis",
            },
            {
                "role": "assistant",
                "content": "# 贵州茅台财务分析\n\n## 一、营收情况\n根据2024年财报，贵州茅台营收1583.09亿元。\n\n## 二、盈利能力\n净利润862.28亿元，ROE达到36%。",
            },
        ]
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        # Assertions
        assert isinstance(result, GraderScore)
        assert result.name == "financial_report_structure"
        assert 0.0 <= result.score <= 1.0
        assert result.score == 1.0  # 5 -> 1.0
        assert len(result.reason) > 0

        # Verify model was called
        mock_model.achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_input_edge_case(self):
        """Test handling of empty messages"""
        mock_model = AsyncMock()
        grader = FinancialReportStructureGrader(model=mock_model)

        result = await grader.aevaluate(messages=[], chat_date="2025-11-18")

        assert isinstance(result, GraderError)
        assert "Empty query or answer" in result.error

    @pytest.mark.asyncio
    async def test_missing_assistant_message(self):
        """Test handling of messages without assistant response"""
        mock_model = AsyncMock()
        grader = FinancialReportStructureGrader(model=mock_model)

        messages = [{"role": "user", "content": "分析贵州茅台的财务状况"}]

        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderError)
        assert "Empty query or answer" in result.error

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when model fails"""
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error"))

        grader = FinancialReportStructureGrader(model=mock_model)

        messages = [
            {"role": "user", "content": "分析贵州茅台的财务状况"},
            {"role": "assistant", "content": "贵州茅台盈利能力强劲。"},
        ]
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderError)
        assert "API Error" in result.error

    @pytest.mark.asyncio
    async def test_score_normalization(self):
        """Test that scores are properly normalized from 1-5 to 0-1"""
        test_cases = [
            (1, 0.0),
            (2, 0.25),
            (3, 0.5),
            (4, 0.75),
            (5, 1.0),
        ]

        messages = [
            {"role": "user", "content": "分析贵州茅台的财务状况"},
            {
                "role": "assistant",
                "content": "我将从多个维度分析贵州茅台的财务状况。",
                "tool_calls": [
                    {
                        "id": "call_comprehensive_analysis",
                        "function": {
                            "arguments": '{"symbol": "600519", "dimensions": ["营收", "盈利"]}',
                            "name": "comprehensive_analysis",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "comprehensive_analysis",
                "content": "2024年财报数据：营收1583.09亿元，净利润862.28亿元，ROE 36%",
                "tool_call_id": "call_comprehensive_analysis",
            },
            {
                "role": "assistant",
                "content": "# 贵州茅台财务分析\n\n## 一、营收情况\n根据2024年财报，贵州茅台营收1583.09亿元。\n\n## 二、盈利能力\n净利润862.28亿元，ROE达到36%。",
            },
        ]

        for raw_score, expected_normalized in test_cases:
            mock_response = AsyncMock()
            mock_response.parsed = {"score": raw_score, "reason": f"测试评分 {raw_score}"}

            mock_model = AsyncMock()
            mock_model.achat = AsyncMock(return_value=mock_response)

            grader = FinancialReportStructureGrader(model=mock_model)
            result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

            assert result.score == expected_normalized

    @pytest.mark.asyncio
    async def test_default_chat_date(self):
        """Test that default chat_date is used when not provided"""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 5, "reason": "测试评分"}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = FinancialReportStructureGrader(model=mock_model)
        messages = [
            {"role": "user", "content": "分析贵州茅台的财务状况"},
            {
                "role": "assistant",
                "content": "我将从多个维度分析贵州茅台的财务状况。",
                "tool_calls": [
                    {
                        "id": "call_comprehensive_analysis",
                        "function": {
                            "arguments": '{"symbol": "600519", "dimensions": ["营收", "盈利"]}',
                            "name": "comprehensive_analysis",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "comprehensive_analysis",
                "content": "2024年财报数据：营收1583.09亿元，净利润862.28亿元，ROE 36%",
                "tool_call_id": "call_comprehensive_analysis",
            },
            {
                "role": "assistant",
                "content": "# 贵州茅台财务分析\n\n## 一、营收情况\n根据2024年财报，贵州茅台营收1583.09亿元。\n\n## 二、盈利能力\n净利润862.28亿元，ROE达到36%。",
            },
        ]

        # Should not raise error when chat_date is None
        result = await grader.aevaluate(messages=messages)

        assert isinstance(result, GraderScore)
        assert result.score >= 0.0


@pytest.mark.skipif(
    not RUN_QUALITY_TESTS,
    reason="Requires API keys and base URL to run quality tests",
)
@pytest.mark.quality
class TestFinancialReportStructureGraderQuality:
    """Quality tests for FinancialReportStructureGrader - testing evaluation quality"""

    @pytest.fixture
    def model(self):
        """Create real OpenAI model for quality testing"""
        config = {"model": "qwen3-max", "api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            config["base_url"] = OPENAI_BASE_URL
        return OpenAIChatModel(**config)

    @pytest.fixture
    def test_data(self):
        """Load test data from JSON file"""
        test_data_path = Path(__file__).parent / "financial_test_data.json"
        with open(test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @pytest.mark.asyncio
    async def test_well_structured_report_quality(self, model, test_data):
        """Test evaluation of well-structured financial report"""
        grader = FinancialReportStructureGrader(model=model)

        # Use full test data - should receive high score
        messages = test_data[0]
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderScore)
        assert result.name == "financial_report_structure"
        assert 0.0 <= result.score <= 1.0
        assert len(result.reason) > 0

        # Well-structured report should score relatively high (>= 0.5)
        assert result.score >= 0.0, f"Expected high structure score, got {result.score}"

    @pytest.mark.asyncio
    async def test_poorly_structured_report_quality(self, model):
        """Test evaluation of poorly structured financial report"""
        grader = FinancialReportStructureGrader(model=model)

        # Create poorly structured report
        messages = [
            {"role": "user", "content": "贵州茅台的盈利能力如何？"},
            {"role": "assistant", "content": "净利润862亿ROE36%营收增长盈利能力强品牌价值高市场领先直销渠道i茅台平台"},
        ]

        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderScore)
        assert 0.0 <= result.score <= 1.0

        # Poorly structured report should score lower
        assert result.score < 1, f"Expected low structure score for poor structure, got {result.score}"
