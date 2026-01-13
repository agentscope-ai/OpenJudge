#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ToolSelectionGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ToolSelectionGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_selection.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_selection.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/tool/test_tool_selection.py -m quality
    ```
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import (
    AccuracyAnalyzer,
    FalseNegativeAnalyzer,
    FalsePositiveAnalyzer,
)
from openjudge.graders.agent import ToolSelectionGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestToolSelectionGraderUnit:
    """Unit tests for ToolSelectionGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ToolSelectionGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "tool_selection"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_good_selection(self):
        """Test successful evaluation with good tool selection"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 4.5,  # Score from 1-5 scale
            "reason": "Appropriate tools selected for the given query",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ToolSelectionGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                query="Find all Python files modified in the last week",
                tool_definitions=[
                    {"name": "search_files", "description": "Search for files"},
                    {"name": "git_log", "description": "Get git history"},
                ],
                tool_calls=[
                    {"name": "search_files", "arguments": {"pattern": "*.py"}},
                    {"name": "git_log", "arguments": {"days": 7}},
                ],
            )

            # Assertions
            assert result.score == 4.5  # Score from 1-5 scale
            assert "appropriate" in result.reason.lower() or "selected" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_poor_selection(self):
        """Test evaluation detecting poor tool selection"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 2.0,  # Score from 1-5 scale
            "reason": "Inappropriate tool selection - selected tools don't match the query intent",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ToolSelectionGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                query="Find all Python files modified in the last week",
                tool_definitions=[
                    {"name": "search_files", "description": "Search for files"},
                    {"name": "read_file", "description": "Read file contents"},
                ],
                tool_calls=[
                    {"name": "read_file", "arguments": {"filepath": "README.md"}},
                ],
            )

            # Assertions
            assert result.score == 2.0  # Score from 1-5 scale
            assert "inappropriate" in result.reason.lower() or "don't match" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ToolSelectionGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                query="test query",
                tool_definitions=[{"name": "test"}],
                tool_calls=[{"name": "test", "arguments": {}}],
            )

            # Assertions
            assert result.score == 0.0
            assert "Evaluation error: API Error" in result.reason


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestToolSelectionGraderQuality:
    """Quality tests for ToolSelectionGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from tool selection pass/fail cases"""
        return [
            # Case 1: Good selection - correct tool for getting sessions
            {
                "query": "I'm interested in joining a virtual yoga or meditation session. Can you provide me with a list of all available sessions?",
                "tool_definitions": [
                    {
                        "name": "Get_All_Sessions",
                        "description": "Get the list of all available yoga and meditation sessions.",
                    },
                    {"name": "Create_New_Session", "description": "Create a new virtual yoga or meditation session."},
                    {
                        "name": "Register_for_Session",
                        "description": "Register for a virtual yoga or meditation session.",
                    },
                ],
                "tool_calls": [{"name": "Get_All_Sessions", "arguments": {}}],
                "human_score": 5.0,  # Good selection
            },
            # Case 2: Poor selection - wrong tool for query
            {
                "query": "I'm interested in joining a virtual yoga or meditation session. Can you provide me with a list of all available sessions?",
                "tool_definitions": [
                    {
                        "name": "Get_All_Sessions",
                        "description": "Get the list of all available yoga and meditation sessions.",
                    },
                    {"name": "Create_New_Session", "description": "Create a new virtual yoga or meditation session."},
                    {
                        "name": "Register_for_Session",
                        "description": "Register for a virtual yoga or meditation session.",
                    },
                ],
                "tool_calls": [{"name": "Register_for_Session", "arguments": {}}],
                "human_score": 1.0,  # Poor selection
            },
            # Case 3: Good selection - get device details
            {
                "query": "Can you give me more details about the 'Blood Pressure Monitor' medical device?",
                "tool_definitions": [
                    {
                        "name": "get_device_compatibility",
                        "description": "API for verifying compatibility of a medical device",
                    },
                    {"name": "add_medical_device", "description": "API for adding a medical device"},
                    {"name": "get_device_details", "description": "API for retrieving details of a medical device"},
                ],
                "tool_calls": [{"name": "get_device_details", "arguments": {"device_name": "Blood Pressure Monitor"}}],
                "human_score": 5.0,
            },
            # Case 4: Poor selection - wrong tool for getting details
            {
                "query": "Can you give me more details about the 'Blood Pressure Monitor' medical device?",
                "tool_definitions": [
                    {
                        "name": "get_device_compatibility",
                        "description": "API for verifying compatibility of a medical device",
                    },
                    {"name": "add_medical_device", "description": "API for adding a medical device"},
                    {"name": "get_device_details", "description": "API for retrieving details of a medical device"},
                ],
                "tool_calls": [
                    {"name": "get_device_compatibility", "arguments": {"device_name": "Blood Pressure Monitor"}},
                ],
                "human_score": 1.0,
            },
            # Case 5: Good selection - schedule appointment
            {
                "query": "I would like to schedule a telehealth video call with Dr. Li at 7 PM on March 12th.",
                "tool_definitions": [
                    {"name": "get_appointments", "description": "Query the list of upcoming appointments"},
                    {"name": "schedule_appointment", "description": "Schedule a new video call appointment"},
                    {"name": "cancel_appointment", "description": "Cancel an existing video call appointment"},
                ],
                "tool_calls": [
                    {
                        "name": "schedule_appointment",
                        "arguments": {
                            "patient_id": "user",
                            "doctor_id": "Dr. Li",
                            "appointment_time": "2039-03-12T19:00:00+08:00",
                        },
                    },
                ],
                "human_score": 5.0,
            },
            # Case 6: Poor selection - no tool selected when needed
            {
                "query": "I would like to schedule a telehealth video call with Dr. Li at 7 PM on March 12th.",
                "tool_definitions": [
                    {"name": "get_appointments", "description": "Query the list of upcoming appointments"},
                    {"name": "schedule_appointment", "description": "Schedule a new video call appointment"},
                    {"name": "cancel_appointment", "description": "Cancel an existing video call appointment"},
                ],
                "tool_calls": [],  # No tool selected
                "human_score": 1.0,
            },
        ]

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen3-max-preview", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_discriminative_power_with_runner(self, dataset, model):
        """Test the grader's ability to distinguish between good and poor tool selection"""
        # Create grader with real model
        grader = ToolSelectionGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "tool_selection": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use AccuracyAnalyzer to calculate accuracy metrics
        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=dataset,
            grader_results=results["tool_selection"],
            label_path="human_score",
        )

        # Assert that quality metrics meet expected thresholds
        assert accuracy_result.accuracy >= 0.6, f"Accuracy below threshold: {accuracy_result.accuracy}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in accuracy_result.metadata
        assert accuracy_result.name == "Accuracy Analysis"

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = ToolSelectionGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "tool_selection_run1": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
                },
            ),
            "tool_selection_run2": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use ConsistencyAnalyzer to calculate consistency metrics
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            dataset=dataset,
            grader_results=results["tool_selection_run1"],
            another_grader_results=results["tool_selection_run2"],
        )

        # Assert that consistency metrics meet expected thresholds
        assert (
            consistency_result.consistency >= 0.7
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestToolSelectionGraderAdversarial:
    """Adversarial tests for ToolSelectionGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with good and poor tool selection pairs"""
        return [
            {
                "query": "Can you give me the address of Dr. Li's clinic? His provider ID is 11223344.",
                "tool_definitions": [
                    {
                        "name": "check_provider_available",
                        "description": "Check the availability of a healthcare provider",
                    },
                    {"name": "book_appointment", "description": "Book an appointment with a healthcare provider"},
                    {"name": "get_provider_info", "description": "Get information about a healthcare provider"},
                ],
                "good_tool_calls": [{"name": "get_provider_info", "arguments": {"provider_id": 11223344}}],
                "poor_tool_calls": [{"name": "check_provider_available", "arguments": {"provider_id": 11223344}}],
                "good_label": 5.0,
                "poor_label": 1.0,
            },
            {
                "query": "I am on 四川北路, near 和平公园. Find nearby hospitals within 2km.",
                "tool_definitions": [
                    {"name": "send_help_alert", "description": "Send a request for emergency medical assistance"},
                    {"name": "get_nearby_hospital", "description": "Find nearby hospitals"},
                    {"name": "get_ambulance_eta", "description": "Get the estimated arrival time of an ambulance"},
                ],
                "good_tool_calls": [
                    {
                        "name": "get_nearby_hospital",
                        "arguments": {"location": "四川北路, near 和平公园", "distance": 2000},
                    },
                ],
                "poor_tool_calls": [
                    {"name": "send_help_alert", "arguments": {"location": "四川北路, near 和平公园", "distance": 2000}},
                ],
                "good_label": 5.0,
                "poor_label": 1.0,
            },
            {
                "query": "I would like to book the 'Yoga Bliss' retreat from March 20th to March 27th.",
                "tool_definitions": [
                    {"name": "search_retreats", "description": "Search for available wellness retreats"},
                    {"name": "book_retreat", "description": "Book a wellness retreat"},
                    {"name": "cancel_booking", "description": "Cancel a booking for a wellness retreat"},
                ],
                "good_tool_calls": [
                    {"name": "search_retreats", "arguments": {"start_date": "2039-03-20", "end_date": "2039-03-27"}},
                ],
                "poor_tool_calls": [],  # No tool selected
                "good_label": 5.0,
                "poor_label": 1.0,
            },
            {
                "query": "Can you tell me the schedule of my prenatal examinations for the next week (March 10-16)?",
                "tool_definitions": [
                    {"name": "check_up_schedule", "description": "Query the schedule of prenatal examinations"},
                    {"name": "nutrition_recommendation", "description": "Get daily nutrition recommendation"},
                    {"name": "baby_kick_count", "description": "Record and query the number of times the baby moves"},
                ],
                "good_tool_calls": [
                    {"name": "check_up_schedule", "arguments": {"start_date": "2039-03-10", "end_date": "2039-03-16"}},
                ],
                "poor_tool_calls": [
                    {"name": "baby_kick_count", "arguments": {"start_date": "2039-03-10", "end_date": "2039-03-16"}},
                ],
                "good_label": 5.0,
                "poor_label": 1.0,
            },
        ]

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "qwen3-max-preview", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_adversarial_tool_selection_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = ToolSelectionGrader(model=model)

        # Configure GraderConfig to evaluate both good and poor selections
        grader_configs = {
            "tool_selection_good": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "good_tool_calls",
                },
            ),
            "tool_selection_poor": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "poor_tool_calls",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for poor selections
        # For 1-5 scale, scores >= 3.0 are considered "positive" (good tool selection)
        fp_analyzer = FalsePositiveAnalyzer(prediction_threshold=3.0)
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["tool_selection_poor"],
            label_path="poor_label",
        )

        # Use FalseNegativeAnalyzer for good selections
        fn_analyzer = FalseNegativeAnalyzer(prediction_threshold=3.0)
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["tool_selection_good"],
            label_path="good_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.5, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.5, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
