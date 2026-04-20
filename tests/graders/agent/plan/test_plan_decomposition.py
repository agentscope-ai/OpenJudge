#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for PlanDecompositionGrader.

Tests plan decomposition evaluation with mocked LLM responses.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/plan/test_plan_decomposition.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/plan/test_plan_decomposition.py -m unit
    ```
"""

from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.agent.plan.plan_decomposition import (
    DEFAULT_PLAN_DECOMPOSITION_TEMPLATE,
    PlanDecompositionGrader,
)
from openjudge.models.schema.prompt_template import LanguageEnum


@pytest.mark.unit
class TestPlanDecompositionGraderUnit:
    """Unit tests for PlanDecompositionGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = PlanDecompositionGrader(model=mock_model)
        assert grader.name == "plan_decomposition"
        assert grader.model == mock_model

    def test_initialization_with_language(self):
        """Test initialization with different languages"""
        mock_model = AsyncMock()
        grader_zh = PlanDecompositionGrader(model=mock_model, language=LanguageEnum.ZH)
        assert grader_zh.language == LanguageEnum.ZH

    def test_default_template_exists(self):
        """Test that default template is properly defined"""
        assert DEFAULT_PLAN_DECOMPOSITION_TEMPLATE is not None
        assert LanguageEnum.EN in DEFAULT_PLAN_DECOMPOSITION_TEMPLATE.messages
        assert LanguageEnum.ZH in DEFAULT_PLAN_DECOMPOSITION_TEMPLATE.messages

    @pytest.mark.asyncio
    async def test_excellent_decomposition(self):
        """Test evaluation of a well-decomposed plan (score=5)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Excellent decomposition. All sub-goals identified, correctly ordered, appropriate granularity.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Book a flight from NYC to London, find a hotel near the airport, and arrange airport transfer",
                plan="1. Search flights NYC→London 2. Book flight 3. Search hotels near London airport 4. Book hotel 5. Search airport transfer options 6. Book transfer",
            )

            assert result.score == 5
            assert "Excellent" in result.reason
            assert result.metadata["raw_score"] == 5
            assert result.metadata["evaluation_type"] == "plan_decomposition"

    @pytest.mark.asyncio
    async def test_good_decomposition(self):
        """Test evaluation of a good plan with minor issues (score=4)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 4,
            "reason": "Good decomposition. All major sub-goals identified with minor ordering issues.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Plan a weekend trip to Paris",
                plan="1. Book flight 2. Find hotel 3. Plan itinerary",
            )

            assert result.score == 4
            assert result.metadata["raw_score"] == 4

    @pytest.mark.asyncio
    async def test_adequate_decomposition(self):
        """Test evaluation of an adequate plan (score=3)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 3,
            "reason": "Adequate decomposition. Most sub-goals identified, but some missing that could affect task completion.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Build a web application with user authentication and payment processing",
                plan="1. Set up project 2. Build frontend 3. Deploy",
            )

            assert result.score == 3
            assert result.metadata["raw_score"] == 3

    @pytest.mark.asyncio
    async def test_poor_decomposition(self):
        """Test evaluation of a poor plan (score=2)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "Poor decomposition. Significant sub-goals missing and incorrect ordering.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Organize a conference for 500 attendees",
                plan="1. Book venue 2. Send invitations",
            )

            assert result.score == 2
            assert result.metadata["raw_score"] == 2

    @pytest.mark.asyncio
    async def test_failed_decomposition(self):
        """Test evaluation of a failed decomposition (score=1)"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "Failed decomposition. The plan does not meaningfully decompose the task.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Plan a complex multi-city European tour",
                plan="Just go to Europe and figure it out.",
            )

            assert result.score == 1
            assert result.metadata["raw_score"] == 1

    @pytest.mark.asyncio
    async def test_with_context(self):
        """Test evaluation with task context"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Excellent decomposition considering the budget constraint.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Plan a vacation to Japan",
                plan="1. Set budget 2. Book flights 3. Book accommodation 4. Plan daily itinerary",
                context="Budget: $3000, Duration: 7 days",
            )

            assert result.score == 5

    @pytest.mark.asyncio
    async def test_with_history(self):
        """Test evaluation with history steps"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 4,
            "reason": "Good decomposition building on previous steps.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            history = [
                {"step": 1, "observation": "Found available flights", "plan": "Search flights"},
                {"step": 2, "observation": "Flight booked", "plan": "Book flight"},
            ]

            result = await grader.aevaluate(
                query="Plan a trip",
                plan="1. Find hotel near airport 2. Book hotel 3. Arrange transfer",
                history=history,
            )

            assert result.score == 4

    @pytest.mark.asyncio
    async def test_with_context_and_history(self):
        """Test evaluation with both context and history"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Excellent decomposition with full context awareness.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            history = [
                {"step": 1, "observation": "Database schema reviewed", "plan": "Review schema"},
            ]

            result = await grader.aevaluate(
                query="Migrate database to new server",
                plan="1. Backup current database 2. Set up new server 3. Migrate data 4. Verify integrity 5. Switch DNS",
                context="Zero-downtime migration required",
                history=history,
            )

            assert result.score == 5

    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Test that metadata contains expected fields"""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 4, "reason": "Good plan."}

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Plan a project",
                plan="1. Requirements 2. Design 3. Implement 4. Test 5. Deploy",
            )

            assert "raw_score" in result.metadata
            assert "evaluation_type" in result.metadata
            assert result.metadata["evaluation_type"] == "plan_decomposition"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when LLM fails"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = PlanDecompositionGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Plan a project",
                plan="1. Step one 2. Step two",
            )

            assert result.score == 0.0
            assert "Evaluation error: API Error" in result.reason
            assert result.metadata["raw_score"] == 0.0
