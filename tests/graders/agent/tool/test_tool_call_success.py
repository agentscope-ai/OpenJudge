#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for ToolCallSuccessGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using ToolCallSuccessGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real data)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_call_success.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_call_success.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/tool/test_tool_call_success.py -m quality
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
from openjudge.graders.agent import ToolCallSuccessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestToolCallSuccessGraderUnit:
    """Unit tests for ToolCallSuccessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = ToolCallSuccessGrader(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader.name == "tool_call_success"
        assert grader.model == mock_model

    @pytest.mark.asyncio
    async def test_successful_evaluation_with_success(self):
        """Test successful evaluation with successful tool calls"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 1.0,
            "reason": "Tool calls executed successfully and returned expected results",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ToolCallSuccessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                tool_definitions=[
                    {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {"location": {"type": "string"}},
                    },
                ],
                tool_calls=[
                    {
                        "name": "get_weather",
                        "arguments": {"location": "New York"},
                    },
                ],
                tool_responses=[
                    "Temperature: 25°C, Condition: sunny",
                ],
            )

            # Assertions
            assert result.score == 1.0
            assert "success" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_with_failure(self):
        """Test evaluation detecting tool call failures"""
        # Setup mock response with the expected metadata structure
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 0.0,
            "reason": "Tool calls failed with error responses",
        }

        # Use patch to mock the model's achat method
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = ToolCallSuccessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                tool_definitions=[
                    {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {"location": {"type": "string"}},
                    },
                ],
                tool_calls=[
                    {
                        "name": "get_weather",
                        "arguments": {"location": "New York"},
                    },
                ],
                tool_responses=[
                    "Error: ConnectionTimeout",
                ],
            )

            # Assertions
            assert result.score == 0.0
            assert "fail" in result.reason.lower() or "error" in result.reason.lower()

            # Verify model was called correctly
            mock_achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Use patch to mock the model's achat method to raise an exception
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = ToolCallSuccessGrader(model=mock_model)

            # Override the model's achat method with our mock
            grader.model.achat = mock_achat

            # Execute test
            result = await grader.aevaluate(
                tool_definitions=[{"name": "test", "description": "test"}],
                tool_calls=[{"name": "test", "arguments": {}}],
                tool_responses=[""],
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
class TestToolCallSuccessGraderQuality:
    """Quality tests for ToolCallSuccessGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load dataset extracted from tool call success/fail cases"""
        return [
            # Case 1: Successful tool call - returns data
            {
                "tool_definitions": [
                    {
                        "name": "Get_All_Sessions",
                        "description": "Get the list of all available yoga and meditation sessions.",
                        "parameters": {},
                    },
                ],
                "tool_calls": [
                    {
                        "name": "Get_All_Sessions",
                        "arguments": {},
                    },
                ],
                "tool_responses": [
                    "Session list retrieved successfully: Hatha yoga on 2023-06-01 with instructor Emily",
                ],
                "human_score": 1,  # Success
            },
            # Case 2: Failed tool call - error response
            {
                "tool_definitions": [
                    {
                        "name": "Get_All_Sessions",
                        "description": "Get the list of all available yoga and meditation sessions.",
                        "parameters": {},
                    },
                ],
                "tool_calls": [
                    {
                        "name": "Get_All_Sessions",
                        "arguments": {},
                    },
                ],
                "tool_responses": [
                    "Error: ResourceNotFound - The requested resource was not found",
                ],
                "human_score": 0,  # Failure
            },
            # Case 3: Successful tool call - device details
            {
                "tool_definitions": [
                    {
                        "name": "get_device_details",
                        "description": "API for retrieving details of a medical device",
                        "parameters": {"device_name": {"type": "string", "required": True}},
                    },
                ],
                "tool_calls": [
                    {
                        "name": "get_device_details",
                        "arguments": {"device_name": "Blood Pressure Monitor"},
                    },
                ],
                "tool_responses": [
                    "Device details: Blood Pressure Monitor, Type: Blood Pressure Monitor, Manufacturer: Omron",
                ],
                "human_score": 1,
            },
            # Case 4: Failed tool call - invalid parameters
            {
                "tool_definitions": [
                    {
                        "name": "get_device_details",
                        "description": "API for retrieving details of a medical device",
                        "parameters": {"device_name": {"type": "string", "required": True}},
                    },
                ],
                "tool_calls": [
                    {
                        "name": "get_device_details",
                        "arguments": {"device_name": "Blood Pressure Monitor"},
                    },
                ],
                "tool_responses": [
                    "Error: InvalidParameters",
                ],
                "human_score": 0,
            },
            # Case 5: Successful appointment scheduling
            {
                "tool_definitions": [
                    {
                        "name": "schedule_appointment",
                        "description": "Schedule a new video call appointment",
                        "parameters": {
                            "patient_id": {"type": "string", "required": True},
                            "doctor_id": {"type": "string", "required": True},
                        },
                    },
                ],
                "tool_calls": [
                    {
                        "name": "schedule_appointment",
                        "arguments": {"patient_id": "user", "doctor_id": "Dr. Li"},
                    },
                ],
                "tool_responses": [
                    "Appointment scheduled successfully with ID: 121232 at 2039-03-12T19:00:00+08:00",
                ],
                "human_score": 1,
            },
            # Case 6: Failed appointment - permission denied
            {
                "tool_definitions": [
                    {
                        "name": "schedule_appointment",
                        "description": "Schedule a new video call appointment",
                        "parameters": {
                            "patient_id": {"type": "string", "required": True},
                            "doctor_id": {"type": "string", "required": True},
                        },
                    },
                ],
                "tool_calls": [
                    {
                        "name": "schedule_appointment",
                        "arguments": {"patient_id": "user", "doctor_id": "Dr. Li"},
                    },
                ],
                "tool_responses": [
                    "Error: PermissionDenied",
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
        """Test the grader's ability to distinguish between successful and failed tool calls"""
        # Create grader with real model
        grader = ToolCallSuccessGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "tool_call_success": GraderConfig(
                grader=grader,
                mapper={
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
                    "tool_responses": "tool_responses",
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
            grader_results=results["tool_call_success"],
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
        grader = ToolCallSuccessGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "tool_call_success_run1": GraderConfig(
                grader=grader,
                mapper={
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
                    "tool_responses": "tool_responses",
                },
            ),
            "tool_call_success_run2": GraderConfig(
                grader=grader,
                mapper={
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "tool_calls",
                    "tool_responses": "tool_responses",
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
            grader_results=results["tool_call_success_run1"],
            another_grader_results=results["tool_call_success_run2"],
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
class TestToolCallSuccessGraderAdversarial:
    """Adversarial tests for ToolCallSuccessGrader - testing robustness"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset with successful and failed tool call pairs"""
        return [
            {
                "tool_definitions": [
                    {"name": "get_provider_info", "description": "Get information about a healthcare provider"},
                ],
                "successful_tool_calls": [
                    {
                        "name": "get_provider_info",
                        "arguments": {"provider_id": 11223344},
                    },
                ],
                "successful_tool_responses": [
                    "Provider info: Dr. Li's Clinic, Specialty: Cardiology, Address: 123 Main St",
                ],
                "failed_tool_calls": [
                    {
                        "name": "get_provider_info",
                        "arguments": {"provider_id": 11223344},
                    },
                ],
                "failed_tool_responses": [
                    "Status: success, but no results found matching your query",
                ],
                "successful_label": 1,
                "failed_label": 0,
            },
            {
                "tool_definitions": [{"name": "get_nearby_hospital", "description": "Find nearby hospitals"}],
                "successful_tool_calls": [
                    {
                        "name": "get_nearby_hospital",
                        "arguments": {"location": "四川北路", "distance": 2000},
                    },
                ],
                "successful_tool_responses": [
                    "Found hospital: 同仁医院 at 长宁区黄桥路170号",
                ],
                "failed_tool_calls": [
                    {
                        "name": "get_nearby_hospital",
                        "arguments": {"location": "四川北路", "distance": 2000},
                    },
                ],
                "failed_tool_responses": [
                    "Error: ServiceUnavailable",
                ],
                "successful_label": 1,
                "failed_label": 0,
            },
            {
                "tool_definitions": [{"name": "daily_journal", "description": "Log moods and stress levels"}],
                "successful_tool_calls": [
                    {
                        "name": "daily_journal",
                        "arguments": {"date": "2039-03-09", "mood_level": 6, "stress_level": 8},
                    },
                ],
                "successful_tool_responses": [
                    "Journal entry successfully logged.",
                ],
                "failed_tool_calls": [
                    {
                        "name": "daily_journal",
                        "arguments": {"date": "2039-03-09", "mood_level": 6, "stress_level": 8},
                    },
                ],
                "failed_tool_responses": [
                    "Error: ResourceNotFound",
                ],
                "successful_label": 1,
                "failed_label": 0,
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
    async def test_adversarial_tool_call_success_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples"""
        # Create grader with real model
        grader = ToolCallSuccessGrader(model=model)

        # Configure GraderConfig to evaluate both successful and failed calls
        grader_configs = {
            "tool_call_success_successful": GraderConfig(
                grader=grader,
                mapper={
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "successful_tool_calls",
                    "tool_responses": "successful_tool_responses",
                },
            ),
            "tool_call_success_failed": GraderConfig(
                grader=grader,
                mapper={
                    "tool_definitions": "tool_definitions",
                    "tool_calls": "failed_tool_calls",
                    "tool_responses": "failed_tool_responses",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset)

        # Use FalsePositiveAnalyzer for failed calls
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=dataset,
            grader_results=results["tool_call_success_failed"],
            label_path="failed_label",
        )

        # Use FalseNegativeAnalyzer for successful calls
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=dataset,
            grader_results=results["tool_call_success_successful"],
            label_path="successful_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.5, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.5, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
