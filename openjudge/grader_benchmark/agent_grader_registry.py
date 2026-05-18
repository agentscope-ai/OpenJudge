# -*- coding: utf-8 -*-
"""
Agent Grader Benchmark Registry.

Maps grader names to their benchmark configurations, specifying
grader class, data file, evaluation mode, and input extraction logic.
"""

from typing import Any, Dict, List, Optional

from openjudge.grader_benchmark.benchmark import GraderBenchmark


def _extract_step_inputs(sample: dict, side: str = "chosen") -> Optional[dict]:
    """Extract inputs for step-level graders (action, memory, observation, plan, reasoning, reflection).

    These graders take individual step components (observation, reflection, plan, action, etc.)
    from the chosen/rejected response.
    """
    input_data = sample.get("input", {})
    response_data = sample.get(side)
    if response_data is None:
        return None

    resp = response_data.get("response", {})
    context = input_data.get("context", {})

    kwargs = {}
    # Map response fields to grader inputs
    for key in ("observation", "reflection", "plan", "action", "memory", "reasoning"):
        if key in resp and resp[key]:
            kwargs[key] = resp[key]

    # Add context and history if available
    if "task_context" in context:
        kwargs["context"] = context["task_context"]
    if "history" in context:
        kwargs["history"] = context["history"]

    if not kwargs:
        return None
    return kwargs


def _extract_tool_inputs(sample: dict, side: str = "chosen") -> Optional[dict]:
    """Extract inputs for tool-based graders."""
    input_data = sample.get("input", {})
    response_data = sample.get(side)
    if response_data is None:
        return None

    resp = response_data.get("response", {})
    context = input_data.get("context", {})

    kwargs = {}
    if "query" in input_data:
        kwargs["query"] = input_data["query"]
    if "tool_definitions" in context:
        kwargs["tool_definitions"] = context["tool_definitions"]
    if "tool_calls" in resp:
        kwargs["tool_calls"] = resp["tool_calls"]
    if "tool_responses" in resp:
        kwargs["tool_responses"] = resp["tool_responses"]

    if not kwargs:
        return None
    return kwargs


def _extract_trajectory_inputs(sample: dict, side: str = "chosen") -> Optional[dict]:
    """Extract inputs for trajectory-based graders (work on full message sequences)."""
    input_data = sample.get("input", {})
    response_data = sample.get(side)
    if response_data is None:
        return None

    resp = response_data.get("response", {})
    context = input_data.get("context", {})

    kwargs = {}
    if "messages" in resp:
        kwargs["messages"] = resp["messages"]
    if "task_context" in context:
        kwargs["context"] = context["task_context"]

    if not kwargs:
        return None
    return kwargs


def _extract_response_inputs(sample: dict, side: str = "chosen") -> Optional[dict]:
    """Extract inputs for response-level graders (query + response)."""
    input_data = sample.get("input", {})
    response_data = sample.get(side)
    if response_data is None:
        return None

    resp = response_data.get("response", {})
    context = input_data.get("context", {})

    kwargs = {}
    if "query" in input_data:
        kwargs["query"] = input_data["query"]
    elif "messages" in input_data:
        kwargs["query"] = input_data["messages"]

    if "response" in resp:
        kwargs["response"] = resp["response"]
    if "task_context" in context:
        kwargs["context"] = context["task_context"]

    if not kwargs:
        return None
    return kwargs


def _extract_messages_inputs(sample: dict, side: str = "chosen") -> Optional[dict]:
    """Extract inputs for message-based graders (action_loop, observation_info_gain, etc.)."""
    input_data = sample.get("input", {})
    response_data = sample.get(side)
    if response_data is None:
        return None

    resp = response_data.get("response", {})

    kwargs = {}
    if "messages" in resp:
        kwargs["messages"] = resp["messages"]
    elif "messages" in input_data:
        kwargs["messages"] = input_data["messages"]

    if not kwargs:
        return None
    return kwargs


def _extract_tool_sequence_inputs(sample: dict, side: str = "chosen") -> Optional[dict]:
    """Extract inputs for tool call sequence match graders (need messages + reference_tool_calls)."""
    input_data = sample.get("input", {})
    response_data = sample.get(side)
    if response_data is None:
        return None

    resp = response_data.get("response", {})
    context = input_data.get("context", {})

    kwargs = {}
    if "messages" in resp:
        kwargs["messages"] = resp["messages"]
    if "reference_tool_calls" in resp:
        kwargs["reference_tool_calls"] = resp["reference_tool_calls"]
    elif "reference_tool_calls" in context:
        kwargs["reference_tool_calls"] = context["reference_tool_calls"]
    if "tool_definitions" in context:
        kwargs["tool_definitions"] = context["tool_definitions"]

    if not kwargs:
        return None
    return kwargs


# Registry of all agent grader benchmark configurations
AGENT_GRADER_REGISTRY: Dict[str, Dict[str, Any]] = {
    # === Action ===
    "action_alignment": {
        "grader_class_import": "openjudge.graders.agent.action.action_alignment:ActionAlignmentGrader",
        "data_file": "action_alignment.json",
        "hf_subdir": "agent/action",
        "eval_mode": "pairwise",
        "expected_accuracy": "88%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "action",
    },
    "action_loop_detection": {
        "grader_class_import": "openjudge.graders.agent.action.action_loop:ActionLoopDetectionGrader",
        "data_file": "action_loop_detection.json",
        "hf_subdir": "agent/action",
        "eval_mode": "pairwise",
        "expected_accuracy": "90%",
        "needs_model": False,
        "grader_kwargs": {"similarity_threshold": 1.0},
        "extract_fn": _extract_messages_inputs,
        "category": "action",
    },
    # === Memory ===
    "memory_accuracy": {
        "grader_class_import": "openjudge.graders.agent.memory.memory_accuracy:MemoryAccuracyGrader",
        "data_file": "memory_accuracy.json",
        "hf_subdir": "agent/memory",
        "eval_mode": "pairwise",
        "expected_accuracy": "78%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "memory",
    },
    "memory_detail_preservation": {
        "grader_class_import": "openjudge.graders.agent.memory.memory_detail_preservation:"
        "MemoryDetailPreservationGrader",
        "data_file": "memory_detail_preservation.json",
        "hf_subdir": "agent/memory",
        "eval_mode": "pairwise",
        "expected_accuracy": "76%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "memory",
    },
    "memory_retrieval_effectiveness": {
        "grader_class_import": "openjudge.graders.agent.memory.memory_retrieval_effectiveness:"
        "MemoryRetrievalEffectivenessGrader",
        "data_file": "memory_retrieval_effectiveness.json",
        "hf_subdir": "agent/memory",
        "eval_mode": "pairwise",
        "expected_accuracy": "100%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "memory",
    },
    # === Observation ===
    "observation_information_gain": {
        "grader_class_import": "openjudge.graders.agent.observation.observation_information_gain:"
        "ObservationInformationGainGrader",
        "data_file": "observation_information_gain.json",
        "hf_subdir": "agent/observation",
        "eval_mode": "pairwise",
        "expected_accuracy": "85%",
        "needs_model": False,
        "extract_fn": _extract_messages_inputs,
        "category": "observation",
    },
    # === Plan ===
    "plan_decomposition": {
        "grader_class_import": "openjudge.graders.agent.plan.plan_decomposition:PlanDecompositionGrader",
        "data_file": "plan_decomposition.json",
        "hf_subdir": "agent/plan",
        "eval_mode": "pairwise",
        "expected_accuracy": "80%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "plan",
    },
    "plan_feasibility": {
        "grader_class_import": "openjudge.graders.agent.plan.plan_feasibility:PlanFeasibilityGrader",
        "data_file": "plan_feasibility.json",
        "hf_subdir": "agent/plan",
        "eval_mode": "pairwise",
        "expected_accuracy": "86%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "plan",
    },
    # === Reasoning ===
    "reasoning_coherence": {
        "grader_class_import": "openjudge.graders.agent.reasoning.reasoning_coherence:ReasoningCoherenceGrader",
        "data_file": "reasoning_coherence.json",
        "hf_subdir": "agent/reasoning",
        "eval_mode": "pairwise",
        "expected_accuracy": "82%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "reasoning",
    },
    "reasoning_groundedness": {
        "grader_class_import": "openjudge.graders.agent.reasoning.reasoning_groundedness:ReasoningGroundednessGrader",
        "data_file": "reasoning_groundedness.json",
        "hf_subdir": "agent/reasoning",
        "eval_mode": "pairwise",
        "expected_accuracy": "80%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "reasoning",
    },
    # === Reflection ===
    "reflection_accuracy": {
        "grader_class_import": "openjudge.graders.agent.reflection.reflection_accuracy:ReflectionAccuracyGrader",
        "data_file": "reflection_accuracy.json",
        "hf_subdir": "agent/reflection",
        "eval_mode": "pairwise",
        "expected_accuracy": "100%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "reflection",
    },
    "reflection_outcome_understanding": {
        "grader_class_import": "openjudge.graders.agent.reflection.reflection_outcome_understanding:"
        "ReflectionOutcomeUnderstandingGrader",
        "data_file": "reflection_outcome_understanding.json",
        "hf_subdir": "agent/reflection",
        "eval_mode": "pairwise",
        "expected_accuracy": "78%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "reflection",
    },
    "reflection_progress_awareness": {
        "grader_class_import": "openjudge.graders.agent.reflection.reflection_progress_awareness:"
        "ReflectionProgressAwarenessGrader",
        "data_file": "reflection_progress_awareness.json",
        "hf_subdir": "agent/reflection",
        "eval_mode": "pairwise",
        "expected_accuracy": "74%",
        "needs_model": True,
        "extract_fn": _extract_step_inputs,
        "category": "reflection",
    },
    # === Response ===
    "response_completeness": {
        "grader_class_import": "openjudge.graders.agent.response.response_completeness:ResponseCompletenessGrader",
        "data_file": "response_completeness.json",
        "hf_subdir": "agent/response",
        "eval_mode": "pairwise",
        "expected_accuracy": "80%",
        "needs_model": True,
        "extract_fn": _extract_response_inputs,
        "category": "response",
    },
    "response_helpfulness": {
        "grader_class_import": "openjudge.graders.agent.response.response_helpfulness:ResponseHelpfulnessGrader",
        "data_file": "response_helpfulness.json",
        "hf_subdir": "agent/response",
        "eval_mode": "pairwise",
        "expected_accuracy": "78%",
        "needs_model": True,
        "extract_fn": _extract_response_inputs,
        "category": "response",
    },
    # === Tool ===
    "tool_selection": {
        "grader_class_import": "openjudge.graders.agent.tool.tool_selection:ToolSelectionGrader",
        "data_file": "tool_selection.json",
        "hf_subdir": "agent/tool",
        "eval_mode": "pairwise",
        "expected_accuracy": "85%",
        "needs_model": True,
        "extract_fn": _extract_tool_inputs,
        "category": "tool",
    },
    "tool_call_accuracy": {
        "grader_class_import": "openjudge.graders.agent.tool.tool_call_accuracy:ToolCallAccuracyGrader",
        "data_file": "tool_call_accuracy.json",
        "hf_subdir": "agent/tool",
        "eval_mode": "pairwise",
        "expected_accuracy": "90%",
        "needs_model": True,
        "extract_fn": _extract_tool_inputs,
        "category": "tool",
    },
    "tool_call_success": {
        "grader_class_import": "openjudge.graders.agent.tool.tool_call_success:ToolCallSuccessGrader",
        "data_file": "tool_call_success.json",
        "hf_subdir": "agent/tool",
        "eval_mode": "pairwise",
        "expected_accuracy": "95%",
        "needs_model": True,
        "extract_fn": _extract_tool_inputs,
        "category": "tool",
    },
    "tool_parameter_check": {
        "grader_class_import": "openjudge.graders.agent.tool.tool_parameter_check:ToolParameterCheckGrader",
        "data_file": "tool_parameter_check.json",
        "hf_subdir": "agent/tool",
        "eval_mode": "pairwise",
        "expected_accuracy": "75%",
        "needs_model": True,
        "extract_fn": _extract_tool_inputs,
        "category": "tool",
    },
    "tool_call_precision_recall_match": {
        "grader_class_import": "openjudge.graders.agent.tool.tool_call_precision_recall_match:"
        "ToolCallPrecisionRecallMatchGrader",
        "data_file": "tool_call_precision_recall_match.json",
        "hf_subdir": "agent/tool",
        "eval_mode": "pairwise",
        "expected_accuracy": "85%",
        "needs_model": False,
        "extract_fn": _extract_tool_inputs,
        "category": "tool",
    },
    "tool_call_step_sequence_match": {
        "grader_class_import": "openjudge.graders.agent.tool.tool_call_step_sequence_match:"
        "ToolCallStepSequenceMatchGrader",
        "data_file": "tool_call_step_sequence_match.json",
        "hf_subdir": "agent/tool",
        "eval_mode": "pairwise",
        "expected_accuracy": "80%",
        "needs_model": False,
        "extract_fn": _extract_tool_sequence_inputs,
        "category": "tool",
    },
    "tool_usage_efficiency": {
        "grader_class_import": "openjudge.graders.agent.tool.tool_usage_efficiency:ToolUsageEfficiencyGrader",
        "data_file": "tool_usage_efficiency.json",
        "hf_subdir": "agent/tool",
        "eval_mode": "pairwise",
        "expected_accuracy": "82%",
        "needs_model": False,
        "extract_fn": _extract_messages_inputs,
        "category": "tool",
    },
    # === Trajectory ===
    "trajectory_accuracy": {
        "grader_class_import": "openjudge.graders.agent.trajectory.trajectory_accuracy:TrajectoryAccuracyGrader",
        "data_file": "trajectory_accuracy.json",
        "hf_subdir": "agent/trajectory",
        "eval_mode": "pairwise",
        "expected_accuracy": "85%",
        "needs_model": True,
        "extract_fn": _extract_trajectory_inputs,
        "category": "trajectory",
    },
    "trajectory_error_recovery": {
        "grader_class_import": "openjudge.graders.agent.trajectory.trajectory_error_recovery:"
        "TrajectoryErrorRecoveryGrader",
        "data_file": "trajectory_error_recovery.json",
        "hf_subdir": "agent/trajectory",
        "eval_mode": "pairwise",
        "expected_accuracy": "80%",
        "needs_model": True,
        "extract_fn": _extract_trajectory_inputs,
        "category": "trajectory",
    },
    "trajectory_step_efficiency": {
        "grader_class_import": "openjudge.graders.agent.trajectory.trajectory_step_efficiency:"
        "TrajectoryStepEfficiencyGrader",
        "data_file": "trajectory_step_efficiency.json",
        "hf_subdir": "agent/trajectory",
        "eval_mode": "pairwise",
        "expected_accuracy": "82%",
        "needs_model": False,
        "extract_fn": _extract_messages_inputs,
        "category": "trajectory",
    },
    "trajectory_comprehensive": {
        "grader_class_import": "openjudge.graders.agent.trajectory.trajectory_comprehensive:"
        "TrajectoryComprehensiveGrader",
        "data_file": "trajectory_comprehensive.json",
        "hf_subdir": "agent/trajectory",
        "eval_mode": "pairwise",
        "expected_accuracy": "80%",
        "needs_model": True,
        "extract_fn": _extract_trajectory_inputs,
        "category": "trajectory",
    },
}


def _import_grader_class(import_path: str):
    """Dynamically import a grader class from a dotted import path.

    Args:
        import_path: Format "module.path:ClassName"

    Returns:
        The imported class.
    """
    module_path, class_name = import_path.rsplit(":", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_benchmark(grader_name: str, **overrides) -> GraderBenchmark:
    """Build a GraderBenchmark instance from the registry.

    Args:
        grader_name: Name of the grader in the registry.
        **overrides: Override any registry config fields.

    Returns:
        A configured GraderBenchmark instance.

    Raises:
        KeyError: If grader_name is not in the registry.
    """
    if grader_name not in AGENT_GRADER_REGISTRY:
        raise KeyError(f"Unknown grader: {grader_name}. " f"Available: {sorted(AGENT_GRADER_REGISTRY.keys())}")

    config = {**AGENT_GRADER_REGISTRY[grader_name], **overrides}
    grader_class = _import_grader_class(config.pop("grader_class_import"))
    extract_fn = config.pop("extract_fn")
    config.pop("category", None)

    benchmark = GraderBenchmark(
        grader_class=grader_class,
        data_file=config.pop("data_file"),
        eval_mode=config.pop("eval_mode", "pairwise"),
        expected_accuracy=config.pop("expected_accuracy", ""),
        needs_model=config.pop("needs_model", True),
        hf_subdir=config.pop("hf_subdir", ""),
        grader_kwargs=config.pop("grader_kwargs", None),
        **config,
    )
    # Override extract_inputs with the custom extraction function
    benchmark.extract_inputs = extract_fn

    return benchmark


def get_graders_by_category(category: str) -> List[str]:
    """Get grader names filtered by category.

    Args:
        category: Category name (e.g., "action", "tool", "trajectory").

    Returns:
        List of grader names in that category.
    """
    return [name for name, config in AGENT_GRADER_REGISTRY.items() if config.get("category") == category]


def get_all_categories() -> List[str]:
    """Get all unique categories."""
    categories = set()
    for config in AGENT_GRADER_REGISTRY.values():
        categories.add(config.get("category", ""))
    return sorted(categories)
