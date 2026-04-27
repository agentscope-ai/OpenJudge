#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ReasoningGroundednessGrader.

Tests reasoning groundedness evaluation with mocked LLM responses.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/reasoning/test_reasoning_groundedness.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/reasoning/test_reasoning_groundedness.py -m unit
    ```
"""

from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.agent.reasoning.reasoning_groundedness import (
    DEFAULT_REASONING_GROUNDEDNESS_TEMPLATE,
    ReasoningGroundednessGrader,
)
from openjudge.models.schema.prompt_template import LanguageEnum


@pytest.mark.unit
class TestReasoningGroundednessGraderUnit:
    """Unit tests for ReasoningGroundednessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ReasoningGroundednessGrader(model=mock_model)
        assert grader.name == "reasoning_groundedness"
        assert grader.model == mock_model

    def test_initialization_with_language(self):
        """Test initialization with different languages"""
        mock_model = AsyncMock()
        grader_zh = ReasoningGroundednessGrader(model=mock_model, language=LanguageEnum.ZH)
        assert grader_zh.language == LanguageEnum.ZH

    def test_default_template_exists(self):
        """Test that default template is properly defined"""
        assert DEFAULT_REASONING_GROUNDEDNESS_TEMPLATE is not None
        assert LanguageEnum.EN in DEFAULT_REASONING_GROUNDEDNESS_TEMPLATE.messages
        assert LanguageEnum.ZH in DEFAULT_REASONING_GROUNDEDNESS_TEMPLATE.messages

    @pytest.mark.asyncio
    async def test_grounded_reasoning(self):
        """Test evaluation of a well-grounded reasoning (score=1.0)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "The reasoning is well-grounded. The inference that a key is needed follows directly from the observation of a locked door.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                observation="The door is locked.",
                reasoning="The door is locked, so I need to find a key to open it.",
            )

            assert result.score == 1.0
            assert "grounded" in result.reason.lower()
            assert result.metadata["raw_score"] == 1.0
            assert result.metadata["evaluation_type"] == "reasoning_groundedness"

    @pytest.mark.asyncio
    async def test_ungrounded_reasoning(self):
        """Test evaluation of ungrounded reasoning with speculation (score=0.0)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.0,
            "reason": "The reasoning contains unsupported speculation. The observation only shows a locked door, but the reasoning assumes the key is hidden under the mat without evidence.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                observation="The door is locked.",
                reasoning="The key must be hidden under the doormat. Landlords always hide spare keys there.",
            )

            assert result.score == 0.0
            assert result.metadata["raw_score"] == 0.0

    @pytest.mark.asyncio
    async def test_score_normalization_high(self):
        """Test that raw scores > 0.5 are normalized to 1.0"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.7,
            "reason": "Mostly grounded with minor unsupported inference.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                observation="The server is down.",
                reasoning="The server is not responding, likely due to a network issue.",
            )

            assert result.score == 1.0
            assert result.metadata["raw_score"] == 0.7

    @pytest.mark.asyncio
    async def test_score_normalization_low(self):
        """Test that raw scores <= 0.5 are normalized to 0.0"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.2,
            "reason": "Reasoning contains significant speculation not supported by observation.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                observation="Error 404 on page load.",
                reasoning="The database was hacked and all data has been stolen by an external attacker.",
            )

            assert result.score == 0.0
            assert result.metadata["raw_score"] == 0.2

    @pytest.mark.asyncio
    async def test_with_context(self):
        """Test evaluation with task context"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "Well-grounded reasoning given the escape room context.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                observation="A painting shows the numbers 1-2-3-4.",
                reasoning="The painting shows numbers 1-2-3-4, which could be a combination for the lock.",
                context="Escape room puzzle: clues are hidden in artworks",
            )

            assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_with_history(self):
        """Test evaluation with history steps"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "Reasoning is grounded in current observation and consistent with history.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            history = [
                {"step": 1, "observation": "Found a locked chest", "reasoning": "Need to find a key"},
                {"step": 2, "observation": "Found a golden key", "reasoning": "This key may open the chest"},
            ]

            result = await grader.aevaluate(
                observation="The golden key fits the lock on the chest.",
                reasoning="The key fits, so I can now open the chest.",
                history=history,
            )

            assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_with_context_and_history(self):
        """Test evaluation with both context and history"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "Fully grounded reasoning with full context awareness.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            history = [
                {"step": 1, "observation": "Server CPU at 95%", "reasoning": "High CPU usage detected"},
            ]

            result = await grader.aevaluate(
                observation="Server response time increased from 200ms to 5000ms.",
                reasoning="The increased response time correlates with the high CPU usage observed earlier.",
                context="Production environment monitoring",
                history=history,
            )

            assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Test that metadata contains expected fields"""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 1.0, "reason": "Grounded reasoning."}

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                observation="The file exists.",
                reasoning="The file exists, so I can read it.",
            )

            assert "raw_score" in result.metadata
            assert "evaluation_type" in result.metadata
            assert result.metadata["evaluation_type"] == "reasoning_groundedness"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when LLM fails"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                observation="Some observation",
                reasoning="Some reasoning",
            )

            assert result.score == 0.0
            assert "Evaluation error: API Error" in result.reason
            assert result.metadata["raw_score"] == 0.0

    @pytest.mark.asyncio
    async def test_hallucinated_details(self):
        """Test evaluation when reasoning contains hallucinated details"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.0,
            "reason": "The reasoning fabricates details not present in the observation. The observation only mentions a red light, but the reasoning claims it's a specific error code.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                observation="A red light is blinking on the device.",
                reasoning="The red light indicates error code E-4021, which means the firmware needs to be reinstalled from version 3.2.1.",
            )

            assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_unsupported_causal_claim(self):
        """Test evaluation when reasoning makes unsupported causal claims"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.0,
            "reason": "Unsupported causal claim: the reasoning assumes the update caused the crash without evidence from the observation.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningGroundednessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                observation="The application crashed after the update was installed.",
                reasoning="The update caused the crash because it introduced a bug in the authentication module.",
            )

            assert result.score == 0.0
