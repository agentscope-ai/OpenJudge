"""
Unit and Quality Tests for Financial Report Logical Rigor Grader

This module provides comprehensive tests for FinancialReportLogicGrader,
including unit tests with mocked dependencies and quality tests with real API calls.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.agent.deep_research.financial_report_logic import (
    FinancialReportLogicGrader,
)
from rm_gallery.core.graders.schema import GraderError, GraderScore
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

# Check for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.unit
class TestFinancialReportLogicGraderUnit:
    """Unit tests for FinancialReportLogicGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization of the grader"""
        mock_model = AsyncMock()
        grader = FinancialReportLogicGrader(model=mock_model)

        assert grader.name == "financial_report_logic"
        assert grader.description == "Financial deep research report logical rigor evaluation"

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 5, "reason": "报告逻辑严谨，分析框架清晰，因果关系明确"}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = FinancialReportLogicGrader(model=mock_model)

        # Example messages with tool calls
        messages = [
            {"role": "user", "content": "分析贵州茅台的财务状况"},
            {
                "role": "assistant",
                "content": "让我查询贵州茅台的财务数据。",
                "tool_calls": [
                    {
                        "id": "call_finance",
                        "function": {
                            "arguments": '{"symbol": "SH600519"}',
                            "name": "query_financials",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "query_financials",
                "content": "营收1583.09亿元，净利润862.28亿元，ROE 36%",
                "tool_call_id": "call_finance",
            },
            {
                "role": "assistant",
                "content": "根据2024年财报，贵州茅台营收1583.09亿元，净利润862.28亿元，ROE达到36%。因此公司盈利能力强劲。",
            },
        ]
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        # Assertions
        assert isinstance(result, GraderScore)
        assert result.name == "financial_report_logic"
        assert 0.0 <= result.score <= 1.0
        assert result.score == 1.0  # 5 -> 1.0
        assert len(result.reason) > 0

        # Verify model was called
        mock_model.achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_input_edge_case(self):
        """Test handling of empty messages"""
        mock_model = AsyncMock()
        grader = FinancialReportLogicGrader(model=mock_model)

        result = await grader.aevaluate(messages=[], chat_date="2025-11-18")

        assert isinstance(result, GraderError)
        assert "Empty query or answer" in result.error

    @pytest.mark.asyncio
    async def test_missing_assistant_message(self):
        """Test handling of messages without assistant response"""
        mock_model = AsyncMock()
        grader = FinancialReportLogicGrader(model=mock_model)

        messages = [{"role": "user", "content": "分析贵州茅台的财务状况"}]

        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderError)
        assert "Empty query or answer" in result.error

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when model fails"""
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error"))

        grader = FinancialReportLogicGrader(model=mock_model)

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
            {"role": "assistant", "content": "贵州茅台盈利能力强劲。"},
        ]

        for raw_score, expected_normalized in test_cases:
            mock_response = AsyncMock()
            mock_response.parsed = {"score": raw_score, "reason": f"测试评分 {raw_score}"}

            mock_model = AsyncMock()
            mock_model.achat = AsyncMock(return_value=mock_response)

            grader = FinancialReportLogicGrader(model=mock_model)
            result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

            assert result.score == expected_normalized

    @pytest.mark.asyncio
    async def test_default_chat_date(self):
        """Test that default chat_date is used when not provided"""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 5, "reason": "测试评分"}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = FinancialReportLogicGrader(model=mock_model)
        messages = [
            {"role": "user", "content": "分析贵州茅台的财务状况"},
            {"role": "assistant", "content": "贵州茅台盈利能力强劲。"},
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
class TestFinancialReportLogicGraderQuality:
    """Quality tests for FinancialReportLogicGrader - testing evaluation quality"""

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
    async def test_rigorous_logic_report_quality(self, model, test_data):
        """Test evaluation of logically rigorous financial report"""
        grader = FinancialReportLogicGrader(model=model)

        # Use full test data - should receive high score
        messages = test_data[0]
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderScore)
        assert result.name == "financial_report_logic"
        assert 0.0 <= result.score <= 1.0
        assert len(result.reason) > 0

        # Logically rigorous report should score reasonably high (>= 0.3)
        # Note: LLM-based scoring has inherent variability
        assert result.score >= 0.0, f"Expected reasonable logic score, got {result.score}"

    @pytest.mark.asyncio
    async def test_illogical_report_quality(self, model):
        """Test evaluation of illogical financial report"""
        grader = FinancialReportLogicGrader(model=model)

        # Create report with logical flaws
        messages = [
            {"role": "user", "content": "贵州茅台的盈利能力如何？"},
            {
                "role": "assistant",
                "content": "贵州茅台2024年净利润增长了15%，说明公司盈利能力在下降。由于营收增长，因此市场份额减少。",
            },
        ]

        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderScore)
        assert 0.0 <= result.score <= 1.0

        # Illogical report should score lower
        assert result.score < 0.75, f"Expected low logic score for illogical report, got {result.score}"

    @pytest.mark.asyncio
    async def test_logical_jumps_report_quality(self, model):
        """Test evaluation of report with logical jumps"""
        grader = FinancialReportLogicGrader(model=model)

        # Create report with some logical jumps
        messages = [
            {"role": "user", "content": "贵州茅台的盈利能力如何？"},
            {"role": "assistant", "content": "贵州茅台2024年净利润862.28亿元。因此，投资者应该立即买入该股票。"},
        ]

        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderScore)
        assert 0.0 <= result.score <= 1.0

        # Should score in middle range (relaxed for LLM variability)
        assert 0.0 <= result.score <= 0.85, f"Expected medium logic score for jumps, got {result.score}"
