# -*- coding: utf-8 -*-
"""Grader factory for creating and running OpenJudge graders."""

import importlib
import inspect
from typing import Any, Optional

from features.grader.config.grader_registry import GRADER_REGISTRY
from shared.utils.helpers import run_async

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderError, GraderScore
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum


def _import_grader_class(class_path: str) -> type:
    """Dynamically import a grader class from its path.

    Args:
        class_path: Full module path to the grader class

    Returns:
        The grader class

    Raises:
        ImportError: If the class cannot be imported
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_grader(
    grader_name: str,
    model: Optional[OpenAIChatModel] = None,
    threshold: float = 0.5,
    language: LanguageEnum = LanguageEnum.EN,
    **extra_params: Any,
) -> BaseGrader:
    """Create a grader instance by name.

    Args:
        grader_name: Name of the grader from GRADER_REGISTRY
        model: OpenAIChatModel instance (required for LLM-based graders)
        threshold: Pass/fail threshold
        language: Language for evaluation prompts
        **extra_params: Additional parameters for specific graders

    Returns:
        Configured grader instance

    Raises:
        ValueError: If grader name is not found
        ImportError: If grader class cannot be imported
    """
    if grader_name not in GRADER_REGISTRY:
        raise ValueError(f"Unknown grader: {grader_name}")

    config = GRADER_REGISTRY[grader_name]
    grader_class = _import_grader_class(config["class_path"])

    # Get the init method's accepted parameters
    try:
        init_params = set(inspect.signature(grader_class).parameters)
    except (ValueError, TypeError):
        init_params = set()

    # Build kwargs based on grader requirements
    kwargs: dict[str, Any] = {}

    # Add model for LLM-based graders
    if config.get("requires_model", False):
        if model is None:
            raise ValueError(f"Grader '{grader_name}' requires a model")
        if "model" in init_params:
            kwargs["model"] = model

    # Add threshold if the grader supports it
    if "threshold" in init_params:
        if config.get("score_range") == (1, 5):
            kwargs["threshold"] = int(threshold)
        else:
            kwargs["threshold"] = threshold

    # Add language for LLM-based graders
    if config.get("requires_model", False) and "language" in init_params:
        kwargs["language"] = language

    # Add extra parameters only if the grader accepts them
    for param_name, param_value in extra_params.items():
        if param_name in init_params:
            kwargs[param_name] = param_value

    return grader_class(**kwargs)


async def _run_evaluation_async(
    grader: BaseGrader,
    **kwargs: Any,
) -> GraderScore | GraderError:
    """Run evaluation asynchronously.

    Args:
        grader: The grader instance
        **kwargs: Evaluation parameters

    Returns:
        GraderScore or GraderError
    """
    return await grader.aevaluate(**kwargs)


def run_evaluation(
    grader: BaseGrader,
    query: str = "",
    response: str = "",
    reference_response: str = "",
    context: str = "",
    **extra_kwargs: Any,
) -> GraderScore | GraderError:
    """Run evaluation synchronously.

    Args:
        grader: The grader instance
        query: User query/prompt
        response: Model response to evaluate
        reference_response: Reference/golden answer
        context: Additional context
        **extra_kwargs: Additional evaluation parameters

    Returns:
        GraderScore or GraderError
    """
    # Build kwargs based on what the grader needs
    kwargs: dict[str, Any] = {}

    if query:
        kwargs["query"] = query
    if response:
        kwargs["response"] = response
    if reference_response:
        kwargs["reference_response"] = reference_response
    if context:
        kwargs["context"] = context

    kwargs.update(extra_kwargs)

    return run_async(_run_evaluation_async(grader, **kwargs))


def run_multimodal_evaluation(
    grader: BaseGrader,
    response_content: list[Any],
    query: str = "",
    **extra_kwargs: Any,
) -> GraderScore | GraderError:
    """Run multimodal evaluation with images.

    Args:
        grader: The grader instance
        response_content: List containing text and MLLMImage objects
        query: Text prompt (for text-to-image evaluation)
        **extra_kwargs: Additional evaluation parameters

    Returns:
        GraderScore or GraderError
    """
    kwargs: dict[str, Any] = {"response": response_content}

    if query:
        kwargs["query"] = query

    kwargs.update(extra_kwargs)

    return run_async(_run_evaluation_async(grader, **kwargs))


def run_agent_evaluation(
    grader: BaseGrader,
    query: str,
    tool_definitions: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    reference_tool_calls: Optional[list[dict[str, Any]]] = None,
    **extra_kwargs: Any,
) -> GraderScore | GraderError:
    """Run agent/tool evaluation.

    Args:
        grader: The grader instance
        query: User query
        tool_definitions: Available tool definitions
        tool_calls: Actual tool calls made by the agent
        reference_tool_calls: Expected tool calls (for accuracy evaluation)
        **extra_kwargs: Additional evaluation parameters

    Returns:
        GraderScore or GraderError
    """
    kwargs: dict[str, Any] = {
        "query": query,
        "tool_definitions": tool_definitions,
        "tool_calls": tool_calls,
    }

    if reference_tool_calls:
        kwargs["reference_tool_calls"] = reference_tool_calls

    kwargs.update(extra_kwargs)

    return run_async(_run_evaluation_async(grader, **kwargs))
