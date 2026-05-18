# -*- coding: utf-8 -*-
"""
Tests for agent_grader_registry module — extraction functions, registry, and build_benchmark.
"""

import pytest

from openjudge.grader_benchmark.agent_grader_registry import (
    AGENT_GRADER_REGISTRY,
    _extract_messages_inputs,
    _extract_response_inputs,
    _extract_step_inputs,
    _extract_tool_inputs,
    _extract_tool_sequence_inputs,
    _extract_trajectory_inputs,
    _import_grader_class,
    build_benchmark,
    get_all_categories,
    get_graders_by_category,
)
from openjudge.grader_benchmark.benchmark import GraderBenchmark
from openjudge.graders.agent.action.action_loop import ActionLoopDetectionGrader

# ============================================================
# _extract_step_inputs
# ============================================================


class TestExtractStepInputs:
    """Tests for _extract_step_inputs."""

    def test_extracts_observation_and_reflection(self):
        sample = {
            "input": {"context": {"task_context": "task info", "history": []}},
            "chosen": {
                "response": {
                    "observation": "saw something",
                    "reflection": "thought about it",
                }
            },
        }
        result = _extract_step_inputs(sample, "chosen")
        assert result is not None
        assert result["observation"] == "saw something"
        assert result["reflection"] == "thought about it"
        assert result["context"] == "task info"
        assert result["history"] == []

    def test_extracts_all_step_fields(self):
        sample = {
            "input": {"context": {"task_context": "ctx"}},
            "chosen": {
                "response": {
                    "observation": "obs",
                    "reflection": "refl",
                    "plan": "pln",
                    "action": "act",
                    "memory": "mem",
                    "reasoning": "reason",
                }
            },
        }
        result = _extract_step_inputs(sample, "chosen")
        assert result is not None
        assert result["observation"] == "obs"
        assert result["reflection"] == "refl"
        assert result["plan"] == "pln"
        assert result["action"] == "act"
        assert result["memory"] == "mem"
        assert result["reasoning"] == "reason"

    def test_skips_empty_fields(self):
        sample = {
            "input": {},
            "chosen": {"response": {"observation": "obs", "plan": ""}},
        }
        result = _extract_step_inputs(sample, "chosen")
        assert result is not None
        assert "observation" in result
        assert "plan" not in result  # empty string is falsy, skipped

    def test_returns_none_when_no_response_data(self):
        sample = {"input": {}, "chosen": None}
        assert _extract_step_inputs(sample, "chosen") is None

    def test_returns_none_when_no_step_fields(self):
        sample = {"input": {}, "chosen": {"response": {}}}
        assert _extract_step_inputs(sample, "chosen") is None

    def test_returns_none_when_response_empty_dict(self):
        sample = {"input": {}, "chosen": {"response": {"observation": ""}}}
        assert _extract_step_inputs(sample, "chosen") is None

    def test_rejected_side(self):
        sample = {
            "input": {},
            "rejected": {"response": {"action": "bad action"}},
        }
        result = _extract_step_inputs(sample, "rejected")
        assert result is not None
        assert result["action"] == "bad action"

    def test_context_without_task_context(self):
        sample = {
            "input": {"context": {"history": ["step1"]}},
            "chosen": {"response": {"action": "act"}},
        }
        result = _extract_step_inputs(sample, "chosen")
        assert result is not None
        assert result["action"] == "act"
        assert result["history"] == ["step1"]
        assert "context" not in result


# ============================================================
# _extract_tool_inputs
# ============================================================


class TestExtractToolInputs:
    """Tests for _extract_tool_inputs."""

    def test_extracts_all_tool_fields(self):
        sample = {
            "input": {
                "query": "search for X",
                "context": {"tool_definitions": [{"name": "search"}]},
            },
            "chosen": {
                "response": {
                    "tool_calls": [{"name": "search", "arguments": '{"q": "X"}'}],
                    "tool_responses": ["result1"],
                }
            },
        }
        result = _extract_tool_inputs(sample, "chosen")
        assert result is not None
        assert result["query"] == "search for X"
        assert result["tool_definitions"] == [{"name": "search"}]
        assert result["tool_calls"] == [{"name": "search", "arguments": '{"q": "X"}'}]
        assert result["tool_responses"] == ["result1"]

    def test_extracts_partial_fields(self):
        sample = {
            "input": {"query": "what?"},
            "chosen": {"response": {"tool_calls": [{"name": "fn"}]}},
        }
        result = _extract_tool_inputs(sample, "chosen")
        assert result is not None
        assert "query" in result
        assert "tool_calls" in result
        assert "tool_definitions" not in result
        assert "tool_responses" not in result

    def test_returns_none_when_no_response_data(self):
        sample = {"input": {}, "chosen": None}
        assert _extract_tool_inputs(sample, "chosen") is None

    def test_returns_none_when_no_tool_fields(self):
        sample = {"input": {}, "chosen": {"response": {}}}
        assert _extract_tool_inputs(sample, "chosen") is None

    def test_rejected_side(self):
        sample = {
            "input": {"query": "q"},
            "rejected": {"response": {"tool_calls": [{"name": "bad_fn"}]}},
        }
        result = _extract_tool_inputs(sample, "rejected")
        assert result is not None
        assert result["tool_calls"] == [{"name": "bad_fn"}]


# ============================================================
# _extract_trajectory_inputs
# ============================================================


class TestExtractTrajectoryInputs:
    """Tests for _extract_trajectory_inputs."""

    def test_extracts_messages_and_context(self):
        sample = {
            "input": {"context": {"task_context": "task desc"}},
            "chosen": {"response": {"messages": [{"role": "user", "content": "hi"}]}},
        }
        result = _extract_trajectory_inputs(sample, "chosen")
        assert result is not None
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["context"] == "task desc"

    def test_extracts_messages_only(self):
        sample = {
            "input": {},
            "chosen": {"response": {"messages": [{"role": "user", "content": "hi"}]}},
        }
        result = _extract_trajectory_inputs(sample, "chosen")
        assert result is not None
        assert "messages" in result
        assert "context" not in result

    def test_returns_none_when_no_response_data(self):
        sample = {"input": {}, "chosen": None}
        assert _extract_trajectory_inputs(sample, "chosen") is None

    def test_returns_none_when_no_messages(self):
        sample = {"input": {}, "chosen": {"response": {}}}
        assert _extract_trajectory_inputs(sample, "chosen") is None

    def test_rejected_side(self):
        sample = {
            "input": {},
            "rejected": {"response": {"messages": [{"role": "assistant", "content": "err"}]}},
        }
        result = _extract_trajectory_inputs(sample, "rejected")
        assert result is not None
        assert result["messages"][0]["content"] == "err"


# ============================================================
# _extract_response_inputs
# ============================================================


class TestExtractResponseInputs:
    """Tests for _extract_response_inputs."""

    def test_extracts_query_and_response(self):
        sample = {
            "input": {"query": "what is X?"},
            "chosen": {"response": {"response": "X is Y"}},
        }
        result = _extract_response_inputs(sample, "chosen")
        assert result is not None
        assert result["query"] == "what is X?"
        assert result["response"] == "X is Y"

    def test_extracts_messages_as_query_fallback(self):
        sample = {
            "input": {"messages": [{"role": "user", "content": "hello"}]},
            "chosen": {"response": {"response": "hi there"}},
        }
        result = _extract_response_inputs(sample, "chosen")
        assert result is not None
        assert result["query"] == [{"role": "user", "content": "hello"}]
        assert result["response"] == "hi there"

    def test_query_takes_precedence_over_messages(self):
        sample = {
            "input": {
                "query": "explicit query",
                "messages": [{"role": "user", "content": "msg query"}],
            },
            "chosen": {"response": {"response": "ans"}},
        }
        result = _extract_response_inputs(sample, "chosen")
        assert result is not None
        assert result["query"] == "explicit query"

    def test_extracts_context(self):
        sample = {
            "input": {"query": "q", "context": {"task_context": "ctx"}},
            "chosen": {"response": {"response": "a"}},
        }
        result = _extract_response_inputs(sample, "chosen")
        assert result is not None
        assert result["context"] == "ctx"

    def test_returns_none_when_no_response_data(self):
        sample = {"input": {}, "chosen": None}
        assert _extract_response_inputs(sample, "chosen") is None

    def test_returns_none_when_no_extractable_fields(self):
        sample = {"input": {}, "chosen": {"response": {}}}
        assert _extract_response_inputs(sample, "chosen") is None

    def test_rejected_side(self):
        sample = {
            "input": {"query": "q"},
            "rejected": {"response": {"response": "bad answer"}},
        }
        result = _extract_response_inputs(sample, "rejected")
        assert result is not None
        assert result["response"] == "bad answer"


# ============================================================
# _extract_messages_inputs
# ============================================================


class TestExtractMessagesInputs:
    """Tests for _extract_messages_inputs."""

    def test_extracts_messages_from_response(self):
        sample = {
            "input": {},
            "chosen": {"response": {"messages": [{"role": "assistant", "content": "hi"}]}},
        }
        result = _extract_messages_inputs(sample, "chosen")
        assert result is not None
        assert result["messages"] == [{"role": "assistant", "content": "hi"}]

    def test_falls_back_to_input_messages(self):
        sample = {
            "input": {"messages": [{"role": "user", "content": "hello"}]},
            "chosen": {"response": {}},
        }
        result = _extract_messages_inputs(sample, "chosen")
        assert result is not None
        assert result["messages"] == [{"role": "user", "content": "hello"}]

    def test_response_messages_takes_precedence(self):
        sample = {
            "input": {"messages": [{"role": "user", "content": "from input"}]},
            "chosen": {"response": {"messages": [{"role": "assistant", "content": "from resp"}]}},
        }
        result = _extract_messages_inputs(sample, "chosen")
        assert result is not None
        assert result["messages"][0]["content"] == "from resp"

    def test_returns_none_when_no_response_data(self):
        sample = {"input": {}, "chosen": None}
        assert _extract_messages_inputs(sample, "chosen") is None

    def test_returns_none_when_no_messages_anywhere(self):
        sample = {"input": {}, "chosen": {"response": {}}}
        assert _extract_messages_inputs(sample, "chosen") is None


# ============================================================
# _extract_tool_sequence_inputs
# ============================================================


class TestExtractToolSequenceInputs:
    """Tests for _extract_tool_sequence_inputs."""

    def test_extracts_all_fields(self):
        sample = {
            "input": {"context": {"tool_definitions": [{"name": "fn"}], "reference_tool_calls": [{"name": "fn"}]}},
            "chosen": {
                "response": {
                    "messages": [{"role": "assistant", "content": "hi"}],
                    "reference_tool_calls": [{"name": "search"}],
                }
            },
        }
        result = _extract_tool_sequence_inputs(sample, "chosen")
        assert result is not None
        assert result["messages"] == [{"role": "assistant", "content": "hi"}]
        # response-level reference_tool_calls takes precedence
        assert result["reference_tool_calls"] == [{"name": "search"}]
        assert result["tool_definitions"] == [{"name": "fn"}]

    def test_reference_tool_calls_from_context_fallback(self):
        sample = {
            "input": {"context": {"reference_tool_calls": [{"name": "ctx_fn"}], "tool_definitions": [{"name": "fn"}]}},
            "chosen": {"response": {"messages": [{"role": "assistant", "content": "hi"}]}},
        }
        result = _extract_tool_sequence_inputs(sample, "chosen")
        assert result is not None
        assert result["reference_tool_calls"] == [{"name": "ctx_fn"}]

    def test_returns_none_when_no_response_data(self):
        sample = {"input": {}, "chosen": None}
        assert _extract_tool_sequence_inputs(sample, "chosen") is None

    def test_returns_none_when_no_extractable_fields(self):
        sample = {"input": {}, "chosen": {"response": {}}}
        assert _extract_tool_sequence_inputs(sample, "chosen") is None


# ============================================================
# _import_grader_class
# ============================================================


class TestImportGraderClass:
    """Tests for _import_grader_class."""

    def test_imports_valid_class(self):
        cls = _import_grader_class("openjudge.graders.agent.action.action_loop:ActionLoopDetectionGrader")
        assert cls is ActionLoopDetectionGrader

    def test_imports_another_class(self):
        from openjudge.graders.agent.observation.observation_information_gain import (
            ObservationInformationGainGrader,
        )

        cls = _import_grader_class(
            "openjudge.graders.agent.observation.observation_information_gain:ObservationInformationGainGrader"
        )
        assert cls is ObservationInformationGainGrader

    def test_raises_on_invalid_module(self):
        with pytest.raises(ModuleNotFoundError):
            _import_grader_class("nonexistent.module:SomeClass")

    def test_raises_on_invalid_class(self):
        with pytest.raises(AttributeError):
            _import_grader_class("openjudge.graders.agent.action.action_loop:NonexistentClass")


# ============================================================
# Registry
# ============================================================


class TestAgentGraderRegistry:
    """Tests for AGENT_GRADER_REGISTRY and query functions."""

    def test_registry_not_empty(self):
        assert len(AGENT_GRADER_REGISTRY) > 0

    def test_registry_required_fields(self):
        required = ["grader_class_import", "data_file", "eval_mode", "needs_model", "category", "extract_fn"]
        for name, config in AGENT_GRADER_REGISTRY.items():
            for field in required:
                assert field in config, f"Missing '{field}' in '{name}'"

    def test_all_grader_classes_importable(self):
        for name, config in AGENT_GRADER_REGISTRY.items():
            cls = _import_grader_class(config["grader_class_import"])
            assert cls is not None, f"Failed to import grader class for '{name}'"

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

    def test_get_graders_by_category_action(self):
        graders = get_graders_by_category("action")
        assert "action_alignment" in graders
        assert "action_loop_detection" in graders

    def test_get_graders_by_category_tool(self):
        graders = get_graders_by_category("tool")
        assert "tool_selection" in graders
        assert "tool_call_accuracy" in graders

    def test_get_graders_by_category_empty(self):
        graders = get_graders_by_category("nonexistent")
        assert graders == []

    def test_get_all_categories_returns_sorted(self):
        categories = get_all_categories()
        assert categories == sorted(categories)


# ============================================================
# build_benchmark
# ============================================================


class TestBuildBenchmark:
    """Tests for build_benchmark."""

    def test_build_rule_based_benchmark(self):
        benchmark = build_benchmark("action_loop_detection")
        assert isinstance(benchmark, GraderBenchmark)
        assert benchmark.needs_model is False
        assert benchmark.grader_class == ActionLoopDetectionGrader

    def test_build_benchmark_extracts_correct_fn(self):
        benchmark = build_benchmark("action_loop_detection")
        # The benchmark should have the messages extraction function assigned
        assert benchmark.extract_inputs is _extract_messages_inputs

    def test_build_benchmark_with_overrides(self):
        benchmark = build_benchmark("action_loop_detection", data_file="custom.json")
        assert benchmark.data_file == "custom.json"

    def test_build_benchmark_unknown_grader(self):
        with pytest.raises(KeyError, match="Unknown grader"):
            build_benchmark("nonexistent_grader")

    def test_build_benchmark_extract_fn_for_step_graders(self):
        benchmark = build_benchmark("action_alignment")
        assert benchmark.extract_inputs is _extract_step_inputs

    def test_build_benchmark_extract_fn_for_tool_graders(self):
        benchmark = build_benchmark("tool_selection")
        assert benchmark.extract_inputs is _extract_tool_inputs

    def test_build_benchmark_extract_fn_for_trajectory_graders(self):
        benchmark = build_benchmark("trajectory_accuracy")
        assert benchmark.extract_inputs is _extract_trajectory_inputs

    def test_build_benchmark_extract_fn_for_response_graders(self):
        benchmark = build_benchmark("response_completeness")
        assert benchmark.extract_inputs is _extract_response_inputs

    def test_build_benchmark_extract_fn_for_tool_sequence_graders(self):
        benchmark = build_benchmark("tool_call_step_sequence_match")
        assert benchmark.extract_inputs is _extract_tool_sequence_inputs

    def test_build_benchmark_grader_kwargs(self):
        benchmark = build_benchmark("action_loop_detection")
        assert benchmark.grader_kwargs == {"similarity_threshold": 1.0}

    def test_build_benchmark_each_category_has_graders(self):
        for category in get_all_categories():
            graders = get_graders_by_category(category)
            assert len(graders) > 0, f"Category '{category}' has no graders"
            for name in graders:
                benchmark = build_benchmark(name)
                assert isinstance(benchmark, GraderBenchmark)
