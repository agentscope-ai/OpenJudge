# -*- coding: utf-8 -*-
"""
Unit tests for ToolUsageEfficiencyGrader.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_usage_efficiency.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/tool/test_tool_usage_efficiency.py -m unit
    ```
"""

import pytest

from openjudge.graders.agent.tool.tool_usage_efficiency import ToolUsageEfficiencyGrader


@pytest.mark.unit
class TestToolUsageEfficiencyGraderUnit:
    """Unit tests for ToolUsageEfficiencyGrader - testing isolated functionality"""

    def test_initialization_default(self):
        """Test initialization with default parameters"""
        grader = ToolUsageEfficiencyGrader()
        assert grader.name == "tool_usage_efficiency"
        assert grader.redundancy_threshold == 0.8
        assert grader.information_gain_threshold == 0.3

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters"""
        grader = ToolUsageEfficiencyGrader(
            redundancy_threshold=0.7,
            information_gain_threshold=0.4,
        )
        assert grader.redundancy_threshold == 0.7
        assert grader.information_gain_threshold == 0.4

    @pytest.mark.asyncio
    async def test_efficient_single_tool_call(self):
        """Test evaluation with a single efficient tool call"""
        grader = ToolUsageEfficiencyGrader()

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

        assert result.score > 0.0
        assert result.metadata["total_tool_calls"] == 1
        assert result.metadata["unique_tool_calls"] == 1
        assert result.metadata["duplicate_calls"] == 0
        assert result.metadata["avg_info_gain"] == 1.0  # First call always has info_gain=1.0

    @pytest.mark.asyncio
    async def test_redundant_tool_calls(self):
        """Test detection of redundant tool calls — same tool with same arguments"""
        grader = ToolUsageEfficiencyGrader()

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
                        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "Sunny, 20°C"},
            {"role": "assistant", "content": "The weather in Beijing is sunny."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.metadata["total_tool_calls"] == 2
        assert result.metadata["duplicate_calls"] == 1
        assert result.metadata["unique_tool_calls"] == 1
        assert result.metadata["redundancy_penalty"] > 0.0

    @pytest.mark.asyncio
    async def test_diverse_tool_calls(self):
        """Test with diverse tool calls — different tools used"""
        grader = ToolUsageEfficiencyGrader()

        messages = [
            {"role": "user", "content": "Get weather and time for Beijing"},
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
                        "function": {"name": "get_time", "arguments": '{"timezone": "Asia/Shanghai"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "name": "get_time", "content": "14:30 CST"},
            {"role": "assistant", "content": "Beijing: sunny 20°C, time 14:30."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.metadata["unique_tools"] == 2
        assert result.metadata["diversity_ratio"] == 1.0
        assert result.metadata["duplicate_calls"] == 0

    @pytest.mark.asyncio
    async def test_low_information_gain(self):
        """Test detection of low information gain — similar observations"""
        grader = ToolUsageEfficiencyGrader(information_gain_threshold=0.3)

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
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "content": "The weather in Beijing is sunny and warm today with 20 degrees",
            },
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
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "get_weather",
                "content": "The weather in Shanghai is sunny and warm today with 20 degrees",
            },
            {"role": "assistant", "content": "Both cities are sunny and warm."},
        ]

        result = await grader.aevaluate(messages=messages)

        # Second observation is very similar to the first → low info gain
        assert result.metadata["low_gain_calls"] >= 1
        assert result.metadata["avg_info_gain"] < 1.0

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test handling when messages are empty"""
        grader = ToolUsageEfficiencyGrader()

        result = await grader.aevaluate(messages=[])

        assert result.score == 0.0
        assert "No action-observation pairs" in result.reason
        assert result.metadata["tool_call_count"] == 0
        assert result.metadata["evaluable"] is False

    @pytest.mark.asyncio
    async def test_messages_without_tool_calls(self):
        """Test handling when messages have no tool calls"""
        grader = ToolUsageEfficiencyGrader()

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.score == 0.0
        assert result.metadata["evaluable"] is False

    @pytest.mark.asyncio
    async def test_messages_wrapped_in_message_key(self):
        """Test handling of messages wrapped in 'message' key"""
        grader = ToolUsageEfficiencyGrader()

        messages = [
            {"message": {"role": "user", "content": "Get weather"}},
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
            {"message": {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny"}},
            {"message": {"role": "assistant", "content": "Sunny in Beijing."}},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.metadata["total_tool_calls"] == 1

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_per_assistant(self):
        """Test handling of multiple tool calls in a single assistant message"""
        grader = ToolUsageEfficiencyGrader()

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

        assert result.metadata["total_tool_calls"] == 2
        assert result.metadata["unique_tool_calls"] == 2
        assert result.metadata["duplicate_calls"] == 0

    @pytest.mark.asyncio
    async def test_score_is_bounded(self):
        """Test that score is always between 0.0 and 1.0"""
        grader = ToolUsageEfficiencyGrader()

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

    def test_extract_tool_signature_valid_json(self):
        """Test _extract_tool_signature with valid JSON arguments"""
        grader = ToolUsageEfficiencyGrader()

        action = {
            "function": {
                "name": "search",
                "arguments": '{"query": "weather", "limit": 5}',
            }
        }

        sig = grader._extract_tool_signature(action)
        assert sig == "search(limit=5,query=weather)"

    def test_extract_tool_signature_empty_args(self):
        """Test _extract_tool_signature with empty arguments"""
        grader = ToolUsageEfficiencyGrader()

        action = {
            "function": {
                "name": "list_files",
                "arguments": "{}",
            }
        }

        sig = grader._extract_tool_signature(action)
        assert sig == "list_files()"

    def test_extract_tool_signature_invalid_json(self):
        """Test _extract_tool_signature with invalid JSON arguments"""
        grader = ToolUsageEfficiencyGrader()

        action = {
            "function": {
                "name": "search",
                "arguments": "not valid json",
            }
        }

        sig = grader._extract_tool_signature(action)
        assert sig == "search(not valid json)"

    def test_extract_tool_signature_dict_args(self):
        """Test _extract_tool_signature when arguments is already a dict"""
        grader = ToolUsageEfficiencyGrader()

        action = {
            "function": {
                "name": "search",
                "arguments": {"query": "weather"},
            }
        }

        sig = grader._extract_tool_signature(action)
        assert sig == "search(query=weather)"

    @pytest.mark.asyncio
    async def test_metadata_fields(self):
        """Test that metadata contains all expected fields"""
        grader = ToolUsageEfficiencyGrader()

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

        expected_keys = [
            "total_tool_calls",
            "unique_tool_calls",
            "duplicate_calls",
            "redundancy_penalty",
            "avg_info_gain",
            "low_gain_calls",
            "info_gains",
            "unique_tools",
            "diversity_ratio",
            "redundancy_threshold",
            "information_gain_threshold",
        ]
        for key in expected_keys:
            assert key in result.metadata, f"Missing metadata key: {key}"

    @pytest.mark.asyncio
    async def test_efficient_vs_redundant_scores(self):
        """Test that efficient trajectories score higher than redundant ones"""
        grader = ToolUsageEfficiencyGrader()

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
            {"role": "assistant", "content": "Sunny."},
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
            {"role": "assistant", "content": "Sunny."},
        ]

        efficient_result = await grader.aevaluate(messages=efficient_messages)
        redundant_result = await grader.aevaluate(messages=redundant_messages)

        assert efficient_result.score > redundant_result.score

    @pytest.mark.asyncio
    async def test_reason_contains_key_metrics(self):
        """Test that reason string contains key evaluation metrics"""
        grader = ToolUsageEfficiencyGrader()

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

        assert "info_gain" in result.reason
        assert "redundancy" in result.reason
        assert "diversity" in result.reason

    @pytest.mark.asyncio
    async def test_same_tool_different_args_not_duplicate(self):
        """Test that same tool with different arguments is not counted as duplicate"""
        grader = ToolUsageEfficiencyGrader()

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
            {"role": "assistant", "content": "Beijing: sunny, Shanghai: rainy."},
        ]

        result = await grader.aevaluate(messages=messages)

        assert result.metadata["duplicate_calls"] == 0
        assert result.metadata["unique_tool_calls"] == 2
