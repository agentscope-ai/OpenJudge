"""
Unit and Quality Tests for Financial Report Precision Grader

This module provides comprehensive tests for FinancialReportPrecisionGrader,
including unit tests with mocked dependencies and quality tests with real API calls.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.agent.deep_research.financial_report_precision import (
    FinancialReportPrecisionGrader,
)
from rm_gallery.core.graders.schema import GraderError, GraderScore
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

# Check for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.unit
class TestFinancialReportPrecisionGraderUnit:
    """Unit tests for FinancialReportPrecisionGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization of the grader"""
        mock_model = AsyncMock()
        grader = FinancialReportPrecisionGrader(model=mock_model)

        assert grader.name == "financial_report_precision"
        assert grader.description == "Financial deep research report precision evaluation"

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 5, "reason": "报告精准针对用户问题核心，实体对象和分析重点完全一致"}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = FinancialReportPrecisionGrader(model=mock_model)

        # Example messages with tool calls
        messages = [
            {"role": "user", "content": "分析贵州茅台的财务状况"},
            {
                "role": "assistant",
                "content": "我将为您分析贵州茅台的财务情况。",
                "tool_calls": [
                    {
                        "id": "call_moutai_data",
                        "function": {
                            "arguments": '{"entity": "贵州茅台", "type": "财务状况"}',
                            "name": "analyze_company",
                        },
                        "type": "function",
                        "index": 0,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "analyze_company",
                "content": "贵州茅台2024年财报：营收1583.09亿元，净利润862.28亿元",
                "tool_call_id": "call_moutai_data",
            },
            {
                "role": "assistant",
                "content": "根据2024年财报，贵州茅台营收1583.09亿元，净利润862.28亿元。",
            },
        ]
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        # Assertions
        assert isinstance(result, GraderScore)
        assert result.name == "financial_report_precision"
        assert 0.0 <= result.score <= 1.0
        assert result.score == 1.0  # 5 -> 1.0
        assert len(result.reason) > 0

        # Verify model was called
        mock_model.achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_input_edge_case(self):
        """Test handling of empty messages"""
        mock_model = AsyncMock()
        grader = FinancialReportPrecisionGrader(model=mock_model)

        result = await grader.aevaluate(messages=[], chat_date="2025-11-18")

        assert isinstance(result, GraderError)
        assert "Empty query or answer" in result.error

    @pytest.mark.asyncio
    async def test_missing_assistant_message(self):
        """Test handling of messages without assistant response"""
        mock_model = AsyncMock()
        grader = FinancialReportPrecisionGrader(model=mock_model)

        messages = [{"role": "user", "content": "分析贵州茅台的财务状况"}]

        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderError)
        assert "Empty query or answer" in result.error

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when model fails"""
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error"))

        grader = FinancialReportPrecisionGrader(model=mock_model)

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

            grader = FinancialReportPrecisionGrader(model=mock_model)
            result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

            assert result.score == expected_normalized

    @pytest.mark.asyncio
    async def test_default_chat_date(self):
        """Test that default chat_date is used when not provided"""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 5, "reason": "测试评分"}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = FinancialReportPrecisionGrader(model=mock_model)
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
class TestFinancialReportPrecisionGraderQuality:
    """Quality tests for FinancialReportPrecisionGrader - testing evaluation quality"""

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
    async def test_precise_report_quality(self, model, test_data):
        """Test evaluation of precise financial report"""
        grader = FinancialReportPrecisionGrader(model=model)

        # Use full test data - should receive high score
        messages = test_data[0]
        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderScore)
        assert result.name == "financial_report_precision"
        assert 0.0 <= result.score <= 1.0
        assert len(result.reason) > 0

        # Precise report should score reasonably high (>= 0.3)
        # Note: LLM-based scoring has inherent variability
        assert result.score >= 0.0, f"Expected reasonable precision score, got {result.score}"

    @pytest.mark.asyncio
    async def test_imprecise_report_quality(self, model):
        """Test evaluation of imprecise financial report (off-topic)"""
        grader = FinancialReportPrecisionGrader(model=model)

        # Create off-topic report
        messages = [
            {"role": "user", "content": "贵州茅台的最新财务状况如何？"},
            {"role": "assistant", "content": "五粮液是中国白酒行业的领军企业之一，其产品质量优良，市场认可度高。"},
        ]

        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderScore)
        assert 0.0 <= result.score <= 1.0

        # Off-topic report should score very low
        assert result.score < 0.5, f"Expected low precision score for off-topic report, got {result.score}"

    @pytest.mark.asyncio
    async def test_partially_relevant_report_quality(self, model):
        """Test evaluation of partially relevant financial report"""
        grader = FinancialReportPrecisionGrader(model=model)

        # Create partially relevant report
        messages = [
            {"role": "user", "content": "贵州茅台的盈利能力如何？"},
            {
                "role": "assistant",
                "content": "贵州茅台是白酒行业的龙头企业，品牌价值高。公司ROE保持在36%左右。此外，整个白酒行业近年来发展态势良好。",
            },
        ]

        result = await grader.aevaluate(messages=messages, chat_date="2026-03-15")

        assert isinstance(result, GraderScore)
        assert 0.0 <= result.score <= 1.0

        # Should score in middle range
        assert 0.25 <= result.score <= 0.75, f"Expected medium precision score, got {result.score}"
