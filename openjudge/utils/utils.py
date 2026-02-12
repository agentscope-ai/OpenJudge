# -*- coding: utf-8 -*-
"""General utility functions.

This module provides various utility functions for common tasks such as
JSON processing, schema manipulation, and data extraction from LLM responses.
"""

import json
from typing import Any, Dict, Optional, Type

from json_repair import repair_json
from loguru import logger
from pydantic import BaseModel


def repair_and_load_json(
    json_str: str,
) -> dict | list | str | float | int | bool | None:
    """The given json_str maybe incomplete, e.g. '{"key', so we need to
    repair and load it into a Python object.

    Args:
        json_str: A potentially incomplete JSON string that needs repair.

    Returns:
        The parsed Python object from the repaired JSON string.

    Raises:
        ValueError: If the JSON string cannot be parsed even after repair.

    Example:
        >>> result = repair_and_load_json('{"key": "value"}')
        >>> print(result)
        {'key': 'value'}
        >>>
        >>> result = repair_and_load_json('{"key"')  # Incomplete JSON
        >>> print(result)  # Repaired and parsed
        {'key': None}
    """
    repaired = json_str
    try:
        repaired = repair_json(json_str)
    except (ValueError, TypeError):
        # repair_json may fail on malformed input, keep original string
        pass

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to decode JSON string `{json_str}` after repairing it " f"into `{repaired}`. Error: {e}",
        ) from e


def _remove_title_field(schema: dict) -> None:
    """Remove the title field from the JSON schema to avoid
    misleading the LLM.

    This function recursively removes 'title' fields from a JSON schema dictionary,
    which helps prevent LLM confusion when using the schema as a tool definition.

    Args:
        schema: The JSON schema dictionary to modify in-place.

    Example:
        >>> schema = {"title": "Test", "properties": {"name": {"title": "Name", "type": "string"}}}
        >>> _remove_title_field(schema)
        >>> print("title" in schema)
        False
        >>> print("title" in schema["properties"]["name"])
        False
    """
    # The top level title field
    if "title" in schema:
        schema.pop("title")

    # properties
    if "properties" in schema:
        for prop in schema["properties"].values():
            if isinstance(prop, dict):
                _remove_title_field(prop)

    # items
    if "items" in schema and isinstance(schema["items"], dict):
        _remove_title_field(schema["items"])

    # additionalProperties
    if "additionalProperties" in schema and isinstance(
        schema["additionalProperties"],
        dict,
    ):
        _remove_title_field(
            schema["additionalProperties"],
        )


def create_tool_from_base_model(
    structured_model: Type[BaseModel],
    tool_name: str = "generate_structured_output",
) -> Dict[str, Any]:
    """Create a function tool definition from a Pydantic BaseModel.

    This function converts a Pydantic BaseModel class into a tool definition
    that can be used with function calling API. The resulting tool
    definition includes the model's JSON schema as parameters, enabling
    structured output generation by forcing the model to call this function
    with properly formatted data.

    Args:
        structured_model: A Pydantic BaseModel class that defines the expected structure
            for the tool's output.
        tool_name: The tool name that used to force the LLM to generate structured
            output by calling this function. Defaults to "generate_structured_output".

    Returns:
        Dict[str, Any]: A tool definition dictionary compatible with
            function calling API, containing type ("function") and
            function dictionary with name, description, and parameters
            (JSON schema).

    Example:
        >>> from pydantic import BaseModel
        >>>
        >>> class PersonInfo(BaseModel):
        ...     name: str
        ...     age: int
        ...     email: str
        ...
        >>> tool = create_tool_from_base_model(PersonInfo, "extract_person")
        >>> print(tool["function"]["name"])
        extract_person
        >>> print(tool["type"])
        function
        >>> print("parameters" in tool["function"])
        True

    Note:
        The function automatically removes the 'title' field from
        the JSON schema to ensure compatibility with function calling
        format. This is handled by the internal `_remove_title_field` function.
    """
    schema = structured_model.model_json_schema()

    _remove_title_field(schema)
    tool_definition = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Generate the required structured output with " "this function",
            "parameters": schema,
        },
    }
    return tool_definition


def trim_and_load_json(response: str, metric: Any = None) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response.

    Handles common cases where LLM wraps JSON in markdown code blocks or text.

    Args:
        response: LLM response string.
        metric: Optional metric instance for error logging.

    Returns:
        Dict[str, Any]: Parsed JSON dictionary.

    Raises:
        ValueError: If JSON cannot be parsed.

    Example:
        >>> response = '''```json
        ... {"score": 8, "reasoning": "Good"}
        ... ```'''
        >>> data = trim_and_load_json(response)
        >>> print(data["score"])
        8
        >>>
        >>> response = '{"name": "Alice", "age": 30}'
        >>> data = trim_and_load_json(response)
        >>> print(data["name"])
        Alice
    """
    # Remove markdown code blocks
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    # Try to parse JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON from response: {e}\nResponse: {response[:200]}"
        if metric:
            metric_name = getattr(metric, "name", "unknown_metric")
            logger.error(f"{metric_name}: {error_msg}")
        raise ValueError(error_msg) from e


def format_conversation_history(
    history: list[Dict[str, str]],
    include_system: bool = False,
) -> str:
    """
    Format conversation history into a readable string.

    This function converts a list of chat messages in ChatMessage format
    into a human-readable string representation for use in evaluation prompts.

    Args:
        history: List of conversation messages in ChatMessage format.
                 Each message should have "role" and "content" keys.
                 Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        include_system: Whether to include system messages. Defaults to False.

    Returns:
        Formatted string representation of the conversation history.
        Returns "<No conversation history>" if history is empty.

    Example:
        >>> history = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ...     {"role": "user", "content": "How are you?"},
        ... ]
        >>> formatted = format_conversation_history(history)
        >>> print(formatted)
        [User]: Hello

        [Assistant]: Hi there!

        [User]: How are you?
    """
    if not history:
        return "<No conversation history>"

    lines = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "system" and not include_system:
            continue
        role_display = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        }.get(role, role)
        lines.append(f"[{role_display}]: {content}")

    return "\n\n".join(lines) if lines else "<No conversation history>"


async def parse_structured_chat_response(
    chat_response: Any,
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Parse structured response from streaming or non-streaming chat response.

    For streaming responses, returns the last chunk's parsed result (complete).
    For non-streaming responses, returns the parsed result directly.

    Args:
        chat_response: Chat response object from model.achat() with structured_model.
            Can be either streaming (async iterator) or non-streaming.
        default: Default dict to return if parsing fails. Defaults to empty dict.

    Returns:
        Dict[str, Any]: The parsed structured response containing fields like
            'score' and 'reason'.

    Example:
        >>> response = await model.achat(messages, structured_model=GraderScoreCallback)
        >>> parsed = await parse_structured_chat_response(response)
        >>> score = parsed.get("score", 5.0)
        >>> reason = parsed.get("reason", "")
    """
    if default is None:
        default = {}

    if hasattr(chat_response, "__aiter__"):
        # Streaming response - only the last chunk contains complete result
        parsed = None
        async for chunk in chat_response:
            if chunk.parsed:
                parsed = chunk.parsed
        return parsed if parsed is not None else default

    # Non-streaming response
    return chat_response.parsed if chat_response.parsed else default
