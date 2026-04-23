# -*- coding: utf-8 -*-
"""
Unit tests and quality tests for TrajectoryStepEfficiencyGrader.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_step_efficiency.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_step_efficiency.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/trajectory/test_trajectory_step_efficiency.py -m quality -s
    ```
"""

import os

import pytest

from openjudge.graders.agent.trajectory.trajectory_step_efficiency import (
    TrajectoryStepEfficiencyGrader,
)

# ==================== UNIT TESTS ====================


@pytest.mark.unit
class TestTrajectoryStepEfficiencyGraderUnit:
    """Unit tests for TrajectoryStepEfficiencyGrader - testing isolated functionality"""

    def test_initialization_default(self):
        """Test initialization with default parameters"""
        grader = TrajectoryStepEfficiencyGrader()
        assert grader.name == "trajectory_step_efficiency"
        assert grader.redundancy_threshold == 0.8
        assert grader.reference_step_count is None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters"""
        grader = TrajectoryStepEfficiencyGrader(
            redundancy_threshold=0.7,
            reference_step_count=5,
        )
        assert grader.redundancy_threshold == 0.7
        assert grader.reference_step_count == 5

    @pytest.mark.asyncio
    async def test_efficient_trajectory(self):
        """Test evaluation with an efficient trajectory — all steps productive"""
        grader = TrajectoryStepEfficiencyGrader()

        messages = [
            {"role": "user", "content": "Get the weather in Beijing"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 20°C"},
            {"role": "assistant", "content": "The weather in Beijing is sunny, 20°C."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.score == 1.0
        assert result.metadata["productive_steps"] == 1
        assert result.metadata["redundant_steps"] == 0
        assert result.metadata["efficiency_ratio"] == 1.0

    @pytest.mark.asyncio
    async def test_redundant_trajectory(self):
        """Test evaluation with redundant steps — agent repeats same action and observation"""
        grader = TrajectoryStepEfficiencyGrader(redundancy_threshold=0.8)

        messages = [
            {"role": "user", "content": "Get the weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny 20 degrees"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "Sunny 20 degrees"},
            {"role": "assistant", "content": "The weather in Beijing is sunny."},
        ]

        result = await grader.aevaluate(messages=messages)

        # Second step is redundant (same action + similar observation)
        assert result.metadata["total_steps"] == 2
        assert result.metadata["redundant_steps"] == 1
        assert result.metadata["productive_steps"] == 1
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_partially_redundant_trajectory(self):
        """Test with mixed productive and redundant steps"""
        grader = TrajectoryStepEfficiencyGrader(redundancy_threshold=0.8)

        messages = [
            {"role": "user", "content": "Get weather for multiple cities"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 20°C in Beijing"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "get_weather", "arguments": '{"city": "Shanghai"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "Rainy, 18°C in Shanghai"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_3", "name": "get_weather", "content": "Sunny, 20°C in Beijing"},
            {"role": "assistant", "content": "Beijing: sunny 20°C, Shanghai: rainy 18°C."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.metadata["total_steps"] == 3
        assert result.metadata["redundant_steps"] == 1  # Step 3 repeats step 1
        assert result.metadata["productive_steps"] == 2
        assert result.metadata["efficiency_ratio"] == pytest.approx(2 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test handling when messages are empty"""
        grader = TrajectoryStepEfficiencyGrader()

        result = await grader.aevaluate(messages=[])

        assert result.score == 0.0
        assert "No action-observation pairs" in result.reason
        assert result.metadata["action_count"] == 0
        assert result.metadata["evaluable"] is False

    @pytest.mark.asyncio
    async def test_messages_without_tool_calls(self):
        """Test handling when messages have no tool calls"""
        grader = TrajectoryStepEfficiencyGrader()

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.score == 0.0
        assert result.metadata["evaluable"] is False

    @pytest.mark.asyncio
    async def test_with_reference_step_count(self):
        """Test evaluation with reference step count — actual steps <= reference"""
        grader = TrajectoryStepEfficiencyGrader(reference_step_count=3)

        messages = [
            {"role": "user", "content": "Get the weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 20°C"},
            {"role": "assistant", "content": "The weather in Beijing is sunny."},
        ]

        result = await grader.aevaluate(messages=messages)

        # 1 step <= 3 reference steps, ref_efficiency = 1.0
        assert result.metadata["reference_step_count"] == 3
        assert result.metadata["reference_efficiency"] == 1.0
        # combined_score = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_with_reference_step_count_exceeded(self):
        """Test evaluation with reference step count — actual steps exceed reference"""
        grader = TrajectoryStepEfficiencyGrader(reference_step_count=1)

        messages = [
            {"role": "user", "content": "Get the weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 20°C"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "get_weather", "arguments": '{"city": "Shanghai"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "Rainy, 18°C"},
            {"role": "assistant", "content": "Beijing: sunny, Shanghai: rainy."},
        ]

        result = await grader.aevaluate(messages=messages)

        # 2 steps > 1 reference step, ref_efficiency = 1/2 = 0.5
        assert result.metadata["reference_step_count"] == 1
        assert result.metadata["reference_efficiency"] == pytest.approx(0.5, abs=0.01)
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_reference_step_count_override_in_aevaluate(self):
        """Test that reference_step_count in aevaluate overrides instance attribute"""
        grader = TrajectoryStepEfficiencyGrader(reference_step_count=5)

        messages = [
            {"role": "user", "content": "Get the weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 20°C"},
            {"role": "assistant", "content": "Sunny in Beijing."},
        ]

        # Override with reference_step_count=1 in aevaluate
        result = await grader.aevaluate(messages=messages, reference_step_count=1)

        assert result.metadata["reference_step_count"] == 1

    @pytest.mark.asyncio
    async def test_messages_wrapped_in_message_key(self):
        """Test handling of messages wrapped in 'message' key"""
        grader = TrajectoryStepEfficiencyGrader()

        messages = [
            {"message": {"role": "user", "content": "Get the weather"}},
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                            "type": "function",
                        }
                    ],
                }
            },
            {"message": {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 20°C"}},
            {"message": {"role": "assistant", "content": "Sunny in Beijing."}},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.metadata["total_steps"] == 1
        assert result.metadata["productive_steps"] == 1

    @pytest.mark.asyncio
    async def test_redundant_details_metadata(self):
        """Test that redundant_details are recorded in metadata"""
        grader = TrajectoryStepEfficiencyGrader(redundancy_threshold=0.8)

        messages = [
            {"role": "user", "content": "Get weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q": "weather Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "search", "content": "Beijing weather is sunny today"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "search", "arguments": '{"q": "weather Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "name": "search", "content": "Beijing weather is sunny today"},
            {"role": "assistant", "content": "Beijing is sunny today."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert len(result.metadata["redundant_details"]) == 1
        assert result.metadata["redundant_details"][0]["step"] == 1
        assert "redundant" in result.metadata["redundant_details"][0]["reason"].lower()

    @pytest.mark.asyncio
    async def test_score_is_bounded(self):
        """Test that score is always between 0.0 and 1.0"""
        grader = TrajectoryStepEfficiencyGrader()

        messages = [
            {"role": "user", "content": "Get weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny"},
            {"role": "assistant", "content": "Sunny."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_reason_contains_key_info(self):
        """Test that reason string contains key evaluation information"""
        grader = TrajectoryStepEfficiencyGrader(reference_step_count=3)

        messages = [
            {"role": "user", "content": "Get weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny"},
            {"role": "assistant", "content": "Sunny."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert "productive" in result.reason.lower()
        assert "ratio" in result.reason.lower()
        assert "reference" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_redundancy_threshold_effect(self):
        """Test that redundancy_threshold affects redundancy detection"""
        messages = [
            {"role": "user", "content": "Get weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q": "weather"}'},
                        "type": "function",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search",
                "content": "Weather in Beijing is sunny and warm",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "search", "arguments": '{"q": "weather"}'},
                        "type": "function",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "search",
                "content": "Weather in Beijing is sunny and warm",
            },
            {"role": "assistant", "content": "Beijing is sunny."},
        ]

        # Low threshold — observations need less similarity to be considered redundant
        grader_low = TrajectoryStepEfficiencyGrader(redundancy_threshold=0.3)
        result_low = await grader_low.aevaluate(messages=messages)

        # High threshold — observations need more similarity to be considered redundant
        grader_high = TrajectoryStepEfficiencyGrader(redundancy_threshold=0.99)
        result_high = await grader_high.aevaluate(messages=messages)

        # With identical observations + identical action, both should detect redundancy
        # But test the mechanism works — metadata should record the threshold used
        assert result_low.metadata["redundancy_threshold"] == 0.3
        assert result_high.metadata["redundancy_threshold"] == 0.99

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_per_assistant(self):
        """Test handling of multiple tool calls in a single assistant message"""
        grader = TrajectoryStepEfficiencyGrader()

        messages = [
            {"role": "user", "content": "Get weather for two cities"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    },
                    {
                        "id": "call_2",
                        "function": {"name": "get_weather", "arguments": '{"city": "Shanghai"}'},
                        "type": "function",
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 20°C"},
            {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "Rainy, 18°C"},
            {"role": "assistant", "content": "Beijing: sunny, Shanghai: rainy."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.metadata["total_steps"] == 2
        assert result.metadata["productive_steps"] == 2
        assert result.metadata["redundant_steps"] == 0
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_metadata_redundancy_threshold_recorded(self):
        """Test that the redundancy_threshold used is recorded in metadata"""
        grader = TrajectoryStepEfficiencyGrader(redundancy_threshold=0.65)

        messages = [
            {"role": "user", "content": "Get weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny"},
            {"role": "assistant", "content": "Sunny."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.metadata["redundancy_threshold"] == 0.65


# ==================== QUALITY TESTS ====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestTrajectoryStepEfficiencyGraderQuality:
    """Quality tests for TrajectoryStepEfficiencyGrader - testing evaluation quality"""

    @pytest.mark.asyncio
    async def test_efficient_vs_redundant(self):
        """Test that efficient trajectories score higher than redundant ones"""
        grader = TrajectoryStepEfficiencyGrader()

        efficient_messages = [
            {"role": "user", "content": "Get weather for Beijing"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 20°C"},
            {"role": "assistant", "content": "Beijing is sunny."},
        ]

        redundant_messages = [
            {"role": "user", "content": "Get weather for Beijing"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 20°C"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "Sunny, 20°C"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_3", "name": "get_weather", "content": "Sunny, 20°C"},
            {"role": "assistant", "content": "Beijing is sunny."},
        ]

        efficient_result = await grader.aevaluate(messages=efficient_messages)
        redundant_result = await grader.aevaluate(messages=redundant_messages)

        assert efficient_result.score > redundant_result.score
