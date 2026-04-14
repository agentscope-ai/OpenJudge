#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for CodeComplexityGrader.

Tests code complexity / over-engineering evaluation with mocked LLM responses.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/code/test_code_complexity.py -v
    ```
"""

from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.code.code_complexity import (
    DEFAULT_CODE_COMPLEXITY_TEMPLATE,
    CodeComplexityGrader,
)
from openjudge.models.schema.prompt_template import LanguageEnum


@pytest.mark.unit
class TestCodeComplexityGraderUnit:
    """Unit tests for CodeComplexityGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = CodeComplexityGrader(model=mock_model)
        assert grader.name == "code_complexity"
        assert grader.threshold == 3
        assert grader.model == mock_model

    def test_initialization_with_custom_threshold(self):
        """Test initialization with custom threshold"""
        mock_model = AsyncMock()
        grader = CodeComplexityGrader(model=mock_model, threshold=4)
        assert grader.threshold == 4

    def test_initialization_invalid_threshold(self):
        """Test initialization with invalid threshold raises ValueError"""
        mock_model = AsyncMock()
        with pytest.raises(ValueError, match="threshold must be in range"):
            CodeComplexityGrader(model=mock_model, threshold=0)

        with pytest.raises(ValueError, match="threshold must be in range"):
            CodeComplexityGrader(model=mock_model, threshold=6)

    def test_initialization_with_language(self):
        """Test initialization with different languages"""
        mock_model = AsyncMock()
        grader_zh = CodeComplexityGrader(model=mock_model, language=LanguageEnum.ZH)
        assert grader_zh.language == LanguageEnum.ZH

    def test_default_template_exists(self):
        """Test that default template is properly defined"""
        assert DEFAULT_CODE_COMPLEXITY_TEMPLATE is not None
        assert LanguageEnum.EN in DEFAULT_CODE_COMPLEXITY_TEMPLATE.messages
        assert LanguageEnum.ZH in DEFAULT_CODE_COMPLEXITY_TEMPLATE.messages

    @pytest.mark.asyncio
    async def test_simple_code_high_score(self):
        """Test evaluation of simple, appropriately complex code"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Complexity perfectly matches the task. The code is clean and concise.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeComplexityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a function that returns the sum of a list of numbers.",
                response="def sum_numbers(numbers):\n    return sum(numbers)",
            )

            assert result.score == 5
            assert "perfectly matches" in result.reason
            assert result.metadata["threshold"] == 3

    @pytest.mark.asyncio
    async def test_over_engineered_code_low_score(self):
        """Test evaluation of over-engineered code"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "Extremely over-engineered. Unnecessary ABC, factory pattern, and class wrapper for a simple sum function.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeComplexityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a function that returns the sum of a list of numbers.",
                response="""from abc import ABC, abstractmethod
from typing import List, Union

class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, values): pass

class SumAggregator(BaseAggregator):
    def aggregate(self, values):
        result = 0
        for value in values:
            result = result + value
        return result

class AggregatorFactory:
    @staticmethod
    def create(strategy="sum"):
        if strategy == "sum":
            return SumAggregator()
        raise ValueError(f"Unknown strategy: {strategy}")

def sum_numbers(numbers):
    factory = AggregatorFactory()
    aggregator = factory.create("sum")
    return aggregator.aggregate(numbers)""",
            )

            assert result.score == 1
            assert "over-engineered" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_moderate_complexity(self):
        """Test evaluation of moderately over-engineered code"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 3,
            "reason": "Noticeably over-engineered. Class wrapper unnecessary for a one-liner function.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeComplexityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Check if a string is a palindrome.",
                response="""class PalindromeChecker:
    def __init__(self, strategy='default'):
        self.strategy = strategy
    def check(self, s):
        cleaned = self._preprocess(s)
        return cleaned == cleaned[::-1]
    def _preprocess(self, s):
        return s.lower().replace(' ', '')""",
            )

            assert result.score == 3

    @pytest.mark.asyncio
    async def test_metadata_contains_threshold(self):
        """Test that metadata contains the threshold value"""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 4, "reason": "Minor redundancy."}

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeComplexityGrader(model=mock_model, threshold=4)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a function.",
                response="def func(): pass",
            )

            assert result.metadata["threshold"] == 4

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when LLM fails"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = CodeComplexityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a function.",
                response="def func(): pass",
            )

            assert "Evaluation error: API Error" in result.error
