# -*- coding: utf-8 -*-
"""
Tests for the GraderHarness base class.
"""

import json
import tempfile
from pathlib import Path

import pytest

from openjudge.evaluation_harness.agent_harness import (
    AGENT_HARNESS_REGISTRY,
    build_harness,
    get_all_categories,
    get_graders_by_category,
)
from openjudge.evaluation_harness.base_harness import EvalResult, GraderHarness
from openjudge.graders.agent.action.action_loop import ActionLoopDetectionGrader
from openjudge.graders.agent.observation.observation_information_gain import (
    ObservationInformationGainGrader,
)


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_eval_result_defaults(self):
        result = EvalResult(grader_name="test", model_name="qwen3-max")
        assert result.accuracy == 0.0
        assert result.correct == 0
        assert result.total == 0
        assert result.status == "pending"
        assert result.results == []

    def test_eval_result_with_values(self):
        result = EvalResult(
            grader_name="test",
            model_name="qwen3-max",
            accuracy=0.85,
            correct=17,
            total=20,
            status="success",
        )
        assert result.accuracy == 0.85
        assert result.correct == 17
        assert result.total == 20


class TestGraderHarness:
    """Tests for GraderHarness base class."""

    def test_init_defaults(self):
        harness = GraderHarness(
            grader_class=ActionLoopDetectionGrader,
            data_file="test.json",
        )
        assert harness.eval_mode == "pairwise"
        assert harness.needs_model is True
        assert harness.expected_accuracy == ""

    def test_init_custom(self):
        harness = GraderHarness(
            grader_class=ActionLoopDetectionGrader,
            data_file="test.json",
            eval_mode="pointwise",
            needs_model=False,
            expected_accuracy="90%",
        )
        assert harness.eval_mode == "pointwise"
        assert harness.needs_model is False
        assert harness.expected_accuracy == "90%"

    def test_create_grader_rule_based(self):
        harness = GraderHarness(
            grader_class=ActionLoopDetectionGrader,
            data_file="test.json",
            needs_model=False,
        )
        grader = harness.create_grader()
        assert isinstance(grader, ActionLoopDetectionGrader)
        assert grader.name == "action_loop_detection"

    def test_create_grader_needs_model(self):
        harness = GraderHarness(
            grader_class=ActionLoopDetectionGrader,
            data_file="test.json",
            needs_model=True,
        )
        with pytest.raises(ValueError, match="Model is required"):
            harness.create_grader()

    @pytest.mark.asyncio
    async def test_load_dataset_local(self):
        # Create a temporary dataset file
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"id": "test_001", "input": {}, "chosen": {"response": {}}, "rejected": {"response": {}}}]
            data_file = Path(tmpdir) / "test_data.json"
            with open(data_file, "w") as f:
                json.dump(data, f)

            harness = GraderHarness(
                grader_class=ActionLoopDetectionGrader,
                data_file="test_data.json",
                needs_model=False,
            )
            result = await harness.load_dataset(local_dir=tmpdir)
            assert len(result) == 1
            assert result[0]["id"] == "test_001"

    @pytest.mark.asyncio
    async def test_evaluate_pairwise_no_loops(self):
        """Test pairwise evaluation with no-loop messages (should score higher)."""
        harness = GraderHarness(
            grader_class=ActionLoopDetectionGrader,
            data_file="test.json",
            needs_model=False,
            grader_kwargs={"similarity_threshold": 1.0},
        )

        # Override extract_inputs for messages-based grader
        def extract_messages(sample, side="chosen"):
            resp = sample.get(side, {})
            if resp is None:
                return None
            messages = resp.get("response", {}).get("messages")
            if messages:
                return {"messages": messages}
            return None

        harness.extract_inputs = extract_messages

        dataset = [
            {
                "id": "test_001",
                "input": {},
                "chosen": {
                    "response": {
                        "messages": [
                            {
                                "role": "assistant",
                                "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "A"}'}}],
                            },
                            {"role": "tool", "content": "Result A"},
                            {
                                "role": "assistant",
                                "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "B"}'}}],
                            },
                            {"role": "tool", "content": "Result B"},
                        ]
                    }
                },
                "rejected": {
                    "response": {
                        "messages": [
                            {
                                "role": "assistant",
                                "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "A"}'}}],
                            },
                            {"role": "tool", "content": "Result A"},
                            {
                                "role": "assistant",
                                "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "A"}'}}],
                            },
                            {"role": "tool", "content": "Result A"},
                        ]
                    }
                },
            }
        ]

        grader = ActionLoopDetectionGrader(similarity_threshold=1.0)
        result = await harness.evaluate_pairwise(grader, dataset)

        assert result.total == 1
        assert result.correct == 1  # Different actions should beat repeated actions
        assert result.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_pairwise_observation_info_gain(self):
        """Test pairwise evaluation with observation info gain grader."""
        harness = GraderHarness(
            grader_class=ObservationInformationGainGrader,
            data_file="test.json",
            needs_model=False,
        )

        def extract_messages(sample, side="chosen"):
            resp = sample.get(side, {})
            if resp is None:
                return None
            messages = resp.get("response", {}).get("messages")
            if messages:
                return {"messages": messages}
            return None

        harness.extract_inputs = extract_messages

        dataset = [
            {
                "id": "test_001",
                "input": {},
                "chosen": {
                    "response": {
                        "messages": [
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {"function": {"name": "search", "arguments": '{"q": "flights to Tokyo"}'}}
                                ],
                            },
                            {"role": "tool", "content": "Found 3 flights to Tokyo on March 15."},
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {"function": {"name": "search", "arguments": '{"q": "hotels in Tokyo"}'}}
                                ],
                            },
                            {"role": "tool", "content": "Found 5 hotels in Shinjuku area."},
                        ]
                    }
                },
                "rejected": {
                    "response": {
                        "messages": [
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {"function": {"name": "search", "arguments": '{"q": "flights to Tokyo"}'}}
                                ],
                            },
                            {"role": "tool", "content": "Found 3 flights to Tokyo on March 15."},
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {"function": {"name": "search", "arguments": '{"q": "flights to Tokyo"}'}}
                                ],
                            },
                            {"role": "tool", "content": "Found 3 flights to Tokyo on March 15."},
                        ]
                    }
                },
            }
        ]

        grader = ObservationInformationGainGrader()
        result = await harness.evaluate_pairwise(grader, dataset)

        assert result.total == 1
        assert result.correct == 1  # Diverse info should beat redundant info

    def test_print_results(self, capsys):
        harness = GraderHarness(
            grader_class=ActionLoopDetectionGrader,
            data_file="test.json",
            needs_model=False,
        )
        result = EvalResult(
            grader_name="test",
            model_name="rule-based",
            accuracy=0.8,
            correct=4,
            total=5,
            results=[
                {"id": "1", "is_correct": True},
                {"id": "2", "is_correct": False, "chosen_score": 0.3, "rejected_score": 0.7},
            ],
        )
        harness.print_results(result)
        captured = capsys.readouterr()
        assert "Error cases" in captured.out
        assert "ID: 2" in captured.out


class TestAgentHarnessRegistry:
    """Tests for the agent harness registry."""

    def test_registry_not_empty(self):
        assert len(AGENT_HARNESS_REGISTRY) > 0

    def test_all_categories_present(self):
        categories = get_all_categories()
        expected = [
            "action",
            "memory",
            "observation",
            "plan",
            "reasoning",
            "reflection",
            "response",
            "tool",
            "trajectory",
        ]
        for cat in expected:
            assert cat in categories, f"Missing category: {cat}"

    def test_get_graders_by_category(self):
        action_graders = get_graders_by_category("action")
        assert "action_alignment" in action_graders
        assert "action_loop_detection" in action_graders

        tool_graders = get_graders_by_category("tool")
        assert "tool_selection" in tool_graders
        assert "tool_usage_efficiency" in tool_graders

    def test_build_harness_rule_based(self):
        harness = build_harness("action_loop_detection")
        assert isinstance(harness, GraderHarness)
        assert harness.needs_model is False
        assert harness.grader_class == ActionLoopDetectionGrader

    def test_build_harness_unknown_grader(self):
        with pytest.raises(KeyError, match="Unknown grader"):
            build_harness("nonexistent_grader")

    def test_registry_config_completeness(self):
        """Verify every registry entry has required fields."""
        required_fields = ["grader_class_import", "data_file", "eval_mode", "needs_model", "category", "extract_fn"]
        for name, config in AGENT_HARNESS_REGISTRY.items():
            for field in required_fields:
                assert field in config, f"Missing field '{field}' in config for '{name}'"

    def test_all_grader_classes_importable(self):
        """Verify all grader classes can be imported."""
        from openjudge.evaluation_harness.agent_harness import _import_grader_class

        for name, config in AGENT_HARNESS_REGISTRY.items():
            import_path = config["grader_class_import"]
            cls = _import_grader_class(import_path)
            assert cls is not None, f"Failed to import {import_path} for grader {name}"
