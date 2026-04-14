#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for CodeBugDetectionGrader.

Tests code bug detection evaluation with mocked LLM responses.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/code/test_code_bug_detection.py -v
    ```
"""

from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.code.code_bug_detection import (
    DEFAULT_CODE_BUG_DETECTION_TEMPLATE,
    CodeBugDetectionGrader,
)
from openjudge.models.schema.prompt_template import LanguageEnum


@pytest.mark.unit
class TestCodeBugDetectionGraderUnit:
    """Unit tests for CodeBugDetectionGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = CodeBugDetectionGrader(model=mock_model)
        assert grader.name == "code_bug_detection"
        assert grader.threshold == 3
        assert grader.model == mock_model

    def test_initialization_with_custom_threshold(self):
        """Test initialization with custom threshold"""
        mock_model = AsyncMock()
        grader = CodeBugDetectionGrader(model=mock_model, threshold=4)
        assert grader.threshold == 4

    def test_initialization_invalid_threshold(self):
        """Test initialization with invalid threshold raises ValueError"""
        mock_model = AsyncMock()
        with pytest.raises(ValueError, match="threshold must be in range"):
            CodeBugDetectionGrader(model=mock_model, threshold=0)

        with pytest.raises(ValueError, match="threshold must be in range"):
            CodeBugDetectionGrader(model=mock_model, threshold=6)

    def test_initialization_with_language(self):
        """Test initialization with different languages"""
        mock_model = AsyncMock()
        grader_zh = CodeBugDetectionGrader(model=mock_model, language=LanguageEnum.ZH)
        assert grader_zh.language == LanguageEnum.ZH

    def test_default_template_exists(self):
        """Test that default template is properly defined"""
        assert DEFAULT_CODE_BUG_DETECTION_TEMPLATE is not None
        # Should have both EN and ZH prompts
        assert LanguageEnum.EN in DEFAULT_CODE_BUG_DETECTION_TEMPLATE.messages
        assert LanguageEnum.ZH in DEFAULT_CODE_BUG_DETECTION_TEMPLATE.messages

    @pytest.mark.asyncio
    async def test_successful_evaluation_no_bugs(self):
        """Test evaluation of bug-free code"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "No bugs detected. The code correctly handles all typical and edge cases.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeBugDetectionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Return the second largest element in a list.",
                response="def second_largest(nums):\n    if len(nums) < 2:\n        raise ValueError('Need at least 2 elements')\n    unique = sorted(set(nums), reverse=True)\n    if len(unique) < 2:\n        raise ValueError('Need at least 2 distinct elements')\n    return unique[1]",
            )

            assert result.score == 5
            assert "No bugs" in result.reason
            assert result.metadata["threshold"] == 3

    @pytest.mark.asyncio
    async def test_successful_evaluation_with_bugs(self):
        """Test evaluation of buggy code"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "Off-by-one on empty list: IndexError when len < 2. Incorrect result for duplicate values.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeBugDetectionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Return the second largest element in a list.",
                response="def second_largest(nums):\n    nums.sort()\n    return nums[-2]",
            )

            assert result.score == 2
            assert "Off-by-one" in result.reason
            assert result.metadata["threshold"] == 3

    @pytest.mark.asyncio
    async def test_critical_bugs(self):
        """Test evaluation of code with critical bugs (score=1)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "Critical bugs: crashes on empty input, infinite loop for negative numbers, returns wrong type.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeBugDetectionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Implement a stack with push, pop, and peek.",
                response="class Stack:\n    def pop(self): return self.data.pop()",
            )

            assert result.score == 1
            assert "Critical" in result.reason

    @pytest.mark.asyncio
    async def test_metadata_contains_threshold(self):
        """Test that metadata contains the threshold value"""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 4, "reason": "Minor potential issues."}

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeBugDetectionGrader(model=mock_model, threshold=4)
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
            grader = CodeBugDetectionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a function.",
                response="def func(): pass",
            )

            assert "Evaluation error: API Error" in result.error
