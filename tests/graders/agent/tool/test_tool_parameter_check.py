#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ToolParameterCheckGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ToolParameterCheckGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_parameter_check.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_parameter_check.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/tool/test_tool_parameter_check.py -m quality
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
from openjudge.graders.agent import ToolParameterCheckGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestToolParameterCheckGraderUnit:
    """Unit tests for ToolParameterCheckGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ToolParameterCheckGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "tool_parameter_check"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_correct_parameters(self):
        """Test successful evaluation with correct parameters"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.9,  # Will be normalized to 1.0 (> 0.5)
            "reason": "Tool parameters are correctly extracted from the query",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ToolParameterCheckGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                query="Search for Python files in src directory",
                tool_definitions=[
                    {
                        "name": "search_files",
                        "parameters": {
                            "pattern": {"type": "string"},
                            "directory": {"type": "string"},
                        },
                    },
                ],
                tool_calls=[
                    {
                        "name": "search_files",
                        "arguments": {"pattern": "*.py", "directory": "src"},
                    },
                ],
            )

            # Assertions
            assert result.score == 1.0  # Normalized from 0.9
            assert "correct" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_hallucinated_parameters(self):
        """Test evaluation detecting hallucinated parameters"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.1,  # Will be normalized to 0.0 (< 0.5)
            "reason": "Tool call contains hallucinated parameters not present in query or tool definition",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ToolParameterCheckGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                query="Search for Python files",
                tool_definitions=[
                    {
                        "name": "search_files",
                        "parameters": {
                            "pattern": {"type": "string"},
                            "directory": {"type": "string"},
                        },
                    },
                ],
                tool_calls=[
                    {
                        "name": "search_files",
                        "arguments": {
                            "pattern": "*.py",
                            "directory": "src",
                            "recursive": True,
                        },  # recursive is hallucinated
                    },
                ],
            )

            # Assertions
            assert result.score == 0.0  # Normalized from 0.1
            assert "hallucinated" in result.reason.lower() or "not present" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ToolParameterCheckGrader(model=mock_model)

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
class TestToolParameterCheckGraderQuality:
    """Quality tests for ToolParameterCheckGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from tool parameter check pass/fail cases"""
        return [
            # Case 1: Correct parameters - all required params from query
            {
                "query": "I'm interested in joining a virtual yoga or meditation session. Can you provide me with a list of all available sessions?",
                "tool_definitions": [
                    {
                        "name": "Get_All_Sessions",
                        "description": "Get the list of all available yoga and meditation sessions.",
                        "parameters": {},
                    },
                ],
                "tool_calls": [{"name": "Get_All_Sessions", "arguments": {}}],
                "human_score": 1,  # Correct
            },
            # Case 2: Missing required parameter
            {
                "query": "I'm interested in joining a virtual yoga or meditation session.",
                "tool_definitions": [
                    {
                        "name": "Create_New_Session",
                        "description": "Create a new virtual yoga or meditation session.",
                        "parameters": {
                            "session_name": {"type": "string", "required": True},
                            "session_instructor": {"type": "string", "required": True},
                        },
                    },
                ],
                "tool_calls": [
                    {
                        "name": "Create_New_Session",
                        "arguments": {"session_name": "Yin yoga"},  # Missing session_instructor
                    },
                ],
                "human_score": 0,  # Incorrect - missing parameter
            },
            # Case 3: Correct parameter extraction
            {
                "query": "Can you give me more details about the 'Blood Pressure Monitor' medical device?",
                "tool_definitions": [
                    {
                        "name": "get_device_details",
                        "description": "API for retrieving details of a medical device",
                        "parameters": {"device_name": {"type": "string", "required": True}},
                    },
                ],
                "tool_calls": [{"name": "get_device_details", "arguments": {"device_name": "Blood Pressure Monitor"}}],
                "human_score": 1,
            },
            # Case 4: Hallucinated/incorrect parameter value
            {
                "query": "Can you give me more details about the 'Blood Pressure Monitor' medical device?",
                "tool_definitions": [
                    {
                        "name": "get_device_details",
                        "description": "API for retrieving details of a medical device",
                        "parameters": {"device_name": {"type": "string", "required": True}},
                    },
                ],
                "tool_calls": [{"name": "get_device_details", "arguments": {"device_name": "incorrect_value_xyz"}}],
                "human_score": 0,
            },
            # Case 5: Correct parameters from conversation context
            {
                "query": "I would like to schedule a telehealth video call with Dr. Li at 7 PM on March 12th.",
                "tool_definitions": [
                    {
                        "name": "schedule_appointment",
                        "description": "Schedule a new video call appointment",
                        "parameters": {
                            "patient_id": {"type": "string", "required": True},
                            "doctor_id": {"type": "string", "required": True},
                            "appointment_time": {"type": "string", "required": True},
                        },
                    },
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
                "human_score": 1,
            },
            # Case 6: Extra hallucinated parameter
            {
                "query": "I would like to schedule a telehealth video call with Dr. Li at 7 PM on March 12th.",
                "tool_definitions": [
                    {
                        "name": "schedule_appointment",
                        "description": "Schedule a new video call appointment",
                        "parameters": {
                            "patient_id": {"type": "string", "required": True},
                            "doctor_id": {"type": "string", "required": True},
                            "appointment_time": {"type": "string", "required": True},
                        },
                    },
                ],
                "tool_calls": [
                    {
                        "name": "schedule_appointment",
                        "arguments": {
                            "patient_id": "user",
                            "doctor_id": "Dr. Li",
                            "appointment_time": "2039-03-12T19:00:00+08:00",
                            "extra_param_not_in_query": "fabricated_value",
                        },
                    },
                ],
                "human_score": 0,
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
        """Test the grader's ability to distinguish between correct and incorrect parameters"""
        # Create grader with real model
        grader = ToolParameterCheckGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "tool_parameter_check": GraderConfig(
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
            grader_results=results["tool_parameter_check"],
            label_path="human_score",
        )

        # Assert that quality metrics meet expected thresholds
        assert accuracy_result.accuracy >= 0.7, f"Accuracy below threshold: {accuracy_result.accuracy}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in accuracy_result.metadata
        assert accuracy_result.name == "Accuracy Analysis"

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = ToolParameterCheckGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "tool_parameter_check_run1": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
                },
            ),
            "tool_parameter_check_run2": GraderConfig(
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
            grader_results=results["tool_parameter_check_run1"],
            another_grader_results=results["tool_parameter_check_run2"],
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
class TestToolParameterCheckGraderAdversarial:
    """Adversarial tests for ToolParameterCheckGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with correct and incorrect parameter pairs"""
        return [
            {
                "query": "Can you give me the address of Dr. Li's clinic? His provider ID is 11223344.",
                "tool_definitions": [
                    {
                        "name": "get_provider_info",
                        "description": "Get information about a healthcare provider.",
                        "parameters": {"provider_id": {"type": "integer", "required": True}},
                    },
                ],
                "correct_tool_calls": [{"name": "get_provider_info", "arguments": {"provider_id": 11223344}}],
                "incorrect_tool_calls": [{"name": "get_provider_info", "arguments": {}}],  # Missing parameter
                "correct_label": 1,
                "incorrect_label": 0,
            },
            {
                "query": "I am on 四川北路, near 和平公园.",
                "tool_definitions": [
                    {
                        "name": "get_nearby_hospital",
                        "description": "Find nearby hospitals",
                        "parameters": {
                            "location": {"type": "string", "required": True},
                            "distance": {"type": "integer", "required": True},
                        },
                    },
                ],
                "correct_tool_calls": [
                    {
                        "name": "get_nearby_hospital",
                        "arguments": {"location": "四川北路, near 和平公园", "distance": 2000},
                    },
                ],
                "incorrect_tool_calls": [
                    {"name": "get_nearby_hospital", "arguments": {"location": 12345, "distance": 2000}},
                ],  # Wrong type
                "correct_label": 1,
                "incorrect_label": 0,
            },
            {
                "query": "I want to check the schedule from March 10 to March 16, 2039.",
                "tool_definitions": [
                    {
                        "name": "check_up_schedule",
                        "description": "Query the schedule of prenatal examinations",
                        "parameters": {
                            "start_date": {"type": "string", "required": True},
                            "end_date": {"type": "string", "required": True},
                        },
                    },
                ],
                "correct_tool_calls": [
                    {"name": "check_up_schedule", "arguments": {"start_date": "2039-03-10", "end_date": "2039-03-16"}},
                ],
                "incorrect_tool_calls": [
                    {"name": "check_up_schedule", "arguments": {"start_date": "2039-03-10", "end_date": 12345}},
                ],  # Wrong type
                "correct_label": 1,
                "incorrect_label": 0,
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
    async def test_adversarial_tool_parameter_check_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = ToolParameterCheckGrader(model=model)

        # Configure GraderConfig to evaluate both correct and incorrect parameters
        grader_configs = {
            "tool_parameter_check_correct": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "correct_tool_calls",
                },
            ),
            "tool_parameter_check_incorrect": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "incorrect_tool_calls",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for incorrect parameters
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["tool_parameter_check_incorrect"],
            label_path="incorrect_label",
        )

        # Use FalseNegativeAnalyzer for correct parameters
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["tool_parameter_check_correct"],
            label_path="correct_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.5, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.5, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
