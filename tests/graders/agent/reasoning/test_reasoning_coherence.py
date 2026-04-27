#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ReasoningCoherenceGrader.

Tests reasoning coherence evaluation with mocked LLM responses.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/reasoning/test_reasoning_coherence.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/reasoning/test_reasoning_coherence.py -m unit
    ```
"""

from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.agent.reasoning.reasoning_coherence import (
    DEFAULT_REASONING_COHERENCE_TEMPLATE,
    ReasoningCoherenceGrader,
)
from openjudge.models.schema.prompt_template import LanguageEnum


@pytest.mark.unit
class TestReasoningCoherenceGraderUnit:
    """Unit tests for ReasoningCoherenceGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ReasoningCoherenceGrader(model=mock_model)
        assert grader.name == "reasoning_coherence"
        assert grader.model == mock_model

    def test_initialization_with_language(self):
        """Test initialization with different languages"""
        mock_model = AsyncMock()
        grader_zh = ReasoningCoherenceGrader(model=mock_model, language=LanguageEnum.ZH)
        assert grader_zh.language == LanguageEnum.ZH

    def test_default_template_exists(self):
        """Test that default template is properly defined"""
        assert DEFAULT_REASONING_COHERENCE_TEMPLATE is not None
        assert LanguageEnum.EN in DEFAULT_REASONING_COHERENCE_TEMPLATE.messages
        assert LanguageEnum.ZH in DEFAULT_REASONING_COHERENCE_TEMPLATE.messages

    @pytest.mark.asyncio
    async def test_coherent_reasoning(self):
        """Test evaluation of a coherent reasoning chain (score=1.0)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "The reasoning chain is logically coherent. The plan directly addresses the reflection, and the action implements the plan.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                reflection="The drawer is locked and I need a key to open it.",
                plan="I will search for the key in the nearby cabinet.",
                action="search cabinet",
            )

            assert result.score == 1.0
            assert "coherent" in result.reason.lower()
            assert result.metadata["raw_score"] == 1.0
            assert result.metadata["evaluation_type"] == "reasoning_coherence"

    @pytest.mark.asyncio
    async def test_incoherent_reasoning(self):
        """Test evaluation of an incoherent reasoning chain (score=0.0)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.0,
            "reason": "The reasoning chain is incoherent. The plan says to search for a key, but the action opens a window instead.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                reflection="The door is locked and I need a key.",
                plan="I will search for the key in the cabinet.",
                action="open window",
            )

            assert result.score == 0.0
            assert result.metadata["raw_score"] == 0.0

    @pytest.mark.asyncio
    async def test_score_normalization_high(self):
        """Test that raw scores > 0.5 are normalized to 1.0"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.8,
            "reason": "Mostly coherent with minor issues.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                reflection="I see a locked door.",
                plan="I should find a key.",
                action="look around",
            )

            assert result.score == 1.0
            assert result.metadata["raw_score"] == 0.8

    @pytest.mark.asyncio
    async def test_score_normalization_low(self):
        """Test that raw scores <= 0.5 are normalized to 0.0"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.3,
            "reason": "Significant logical gaps in the reasoning chain.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                reflection="The file is corrupted.",
                plan="I will delete the database.",
                action="send email",
            )

            assert result.score == 0.0
            assert result.metadata["raw_score"] == 0.3

    @pytest.mark.asyncio
    async def test_with_context(self):
        """Test evaluation with task context"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "Coherent reasoning given the escape room context.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                reflection="The chest is locked with a combination lock.",
                plan="I will try the numbers from the painting on the wall.",
                action="unlock chest with code 1234",
                context="Escape room: The painter's birthday is on the wall",
            )

            assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_with_history(self):
        """Test evaluation with history steps"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "Coherent reasoning building on previous steps.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            history = [
                {"step": 1, "observation": "Found a locked door", "plan": "Search for key"},
                {"step": 2, "observation": "Found a key on the table", "plan": "Use key on door"},
            ]

            result = await grader.aevaluate(
                reflection="The key from the table should open this door.",
                plan="I will use the key to unlock the door.",
                action="use key on door",
                history=history,
            )

            assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_with_context_and_history(self):
        """Test evaluation with both context and history"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "Fully coherent reasoning with context awareness.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            history = [
                {"step": 1, "observation": "Server returning 500 errors", "plan": "Check logs"},
            ]

            result = await grader.aevaluate(
                reflection="The logs show a database connection timeout.",
                plan="I will restart the database service.",
                action="restart database",
                context="Production environment, maintenance window active",
                history=history,
            )

            assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Test that metadata contains expected fields"""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 1.0, "reason": "Coherent reasoning."}

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                reflection="I need to find a key.",
                plan="Search the desk.",
                action="search desk",
            )

            assert "raw_score" in result.metadata
            assert "evaluation_type" in result.metadata
            assert result.metadata["evaluation_type"] == "reasoning_coherence"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when LLM fails"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                reflection="Some reflection",
                plan="Some plan",
                action="Some action",
            )

            assert result.score == 0.0
            assert "Evaluation error: API Error" in result.reason
            assert result.metadata["raw_score"] == 0.0

    @pytest.mark.asyncio
    async def test_contradictory_plan_action(self):
        """Test evaluation when plan and action contradict"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.0,
            "reason": "Contradiction: plan says to search A but action searches B.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                reflection="I need to find the document in the filing cabinet.",
                plan="I will search the filing cabinet for the document.",
                action="search bookshelf",
            )

            assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_missing_step_in_chain(self):
        """Test evaluation when there's a logical gap in the reasoning chain"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.0,
            "reason": "Logical gap: the reflection identifies a locked door, but the plan and action skip finding a key.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ReasoningCoherenceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                reflection="The door is locked and requires a key.",
                plan="I will open the door.",
                action="open door",
            )

            assert result.score == 0.0
