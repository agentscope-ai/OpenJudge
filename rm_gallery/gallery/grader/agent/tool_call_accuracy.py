# -*- coding: utf-8 -*-
"""Tool call accuracy grader for evaluating agent tool usage."""

import json
import re
from typing import Any, Dict, List, Union

from rm_gallery.core.grader.base import GraderScore, LLMGrader, GraderMode
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template, PromptDict


TOOL_CALL_ACCURACY_SYSTEM_PROMPT = """# Instruction
## Goal
Your are an expert in evaluating the accuracy of a tool call considering relevance and potential usefulness including syntactic and semantic correctness of a proposed tool call from an intelligent system based on provided definition and data. Your goal will involve answering the questions below using the information provided.

# Definition
**Tool Call Accuracy** refers to the overall effectiveness of ALL TOOL CALLS made by an agent in response to a user's query within an ongoing CONVERSATION.

# EVALUATION CRITERIA
Evaluate based on these factors:
1. **Collective Relevance**: Do the tool calls, taken together, appropriately address the user's query?
2. **Parameter Correctness**: Are all parameter values extracted from or reasonably inferred from the CONVERSATION?
   - Fabricated parameters automatically result in Level 2
3. **Completeness**: Did the agent make all necessary tool calls available in the tool definitions?
   - Failed calls don't count as missing
4. **Efficiency**: Did the agent avoid unnecessary duplicate tool calls with identical parameters?
   - Don't penalize single tools returning multiple results (like file_search)
5. **Execution Success**: Were tool calls executed successfully or recovered from errors appropriately?
6. **Scope Limitation**: ONLY evaluate tool calls in the "TOOL CALLS TO BE EVALUATED" section.
   - Tool calls in the CONVERSATION section are for context only
   - Focus exclusively on the agent's response to the user's LAST query
   - Use conversation history only to verify parameter correctness and context

# Ratings
## [Tool Call Accuracy: 1] (Irrelevant)
Tool calls were not relevant to the user's query, resulting in an irrelevant or unhelpful final output.

## [Tool Call Accuracy: 2] (Partially Relevant - Wrong Execution)
Tool calls were somewhat related to the user's query, but the agent was not able to reach information that helps address the user query due to one or more of the following:
  • Parameters passed to the tool were incorrect.
  • Not enough tools (available in the tool definitions) were called to fully help address the query (missing tool calls).
  • Tools returned errors, and no retrials for the tool call were successful.

## [Tool Call Accuracy: 3] (Relevant but Inefficient)
Tool calls were relevant, correct and grounded parameters were passed so that led to a correct output. However, multiple excessive, unnecessary tool calls were made.

## [Tool Call Accuracy: 4] (Correct with Retrials)
Tool calls were fully relevant and efficient:
• Correct tools were called with the correct and grounded parameters, whether they are extracted from the conversation history or the current user query.
• A tool returned an error, but the agent retried calling the tool and successfully got an output.

## [Tool Call Accuracy: 5] (Optimal Solution)
Tool calls were fully relevant and efficient:
  • Correct tools were called with the correct and grounded parameters
  • No unnecessary or excessive tool calls were made
  • No errors occurred in any of the tools
  • The tool calls made helped the agent address the user's query without facing any issues"""


TOOL_CALL_ACCURACY_USER_PROMPT = """# Data
CONVERSATION : {query}
TOOL CALLS TO BE EVALUATED: {tool_calls}
TOOL DEFINITIONS: {tool_definitions}

# Tasks
Please provide your evaluation for the tool calls in relation to the user query and tool definitions.
Your output should be a JSON object with the following format:
```json
{{
    "score": [Tool Call Accuracy Score],
    "reason": [Reason for the score],
}}
```
"""


class ToolCallAccuracyGrader(LLMGrader):
    """Evaluates the accuracy of tool calls made by an agent.

    The ToolCallAccuracyGrader assesses how accurately an AI uses tools by examining:
    - Relevance to the conversation
    - Parameter correctness according to tool definitions
    - Parameter value extraction from the conversation

    The evaluator uses a scoring rubric of 1 to 5:
    - Score 1: The tool calls are irrelevant
    - Score 2: The tool calls are partially relevant, but not enough tools were called or the parameters were not correctly passed
    - Score 3: The tool calls are relevant, but there were unnecessary, excessive tool calls made
    - Score 4: The tool calls are relevant, but some tools returned errors and agent retried calling them again and succeeded
    - Score 5: The tool calls are relevant, and all parameters were correctly passed

    This evaluation focuses on measuring whether tool calls meaningfully contribute to addressing
    user needs while properly following tool definitions and using information present in the
    conversation history.

    Example:
        >>> import asyncio
        >>> grader = ToolCallAccuracyGrader()
        >>> conversation = [
        ...     {"role": "user", "content": "What's the weather like in New York?"}
        ... ]
        >>> tool_definitions = [
        ...     {
        ...         "name": "get_weather",
        ...         "description": "Get weather information for a location",
        ...         "parameters": {
        ...             "location": "City name"
        ...         }
        ...     }
        ... ]
        >>> tool_calls = [
        ...     {
        ...         "name": "get_weather",
        ...         "arguments": {"location": "New York"},
        ...         "result": {"temperature": 25, "condition": "sunny"}
        ...     }
        ... ]
        >>> result = asyncio.run(grader.evaluate(
        ...     query=conversation,
        ...     tool_definitions=tool_definitions,
        ...     tool_calls=tool_calls
        ... ))
        >>> print(result.score)
        5.0
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        **kwargs,
    ):
        """Initialize the ToolCallAccuracyGrader.

        Args:
            model: The language model used for evaluation. Can be either a ChatModelBase
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="tool_call_accuracy",
            mode=GraderMode.POINTWISE,
            description="Evaluates the accuracy of tool calls made by an agent",
            template=Template(
                messages=[
                    ChatMessage(
                        role="system",
                        content=TOOL_CALL_ACCURACY_SYSTEM_PROMPT,
                    ),
                    ChatMessage(
                        role="user",
                        content=TOOL_CALL_ACCURACY_USER_PROMPT,
                    ),
                ],
            ),
            model=model,
            **kwargs,
        )

    def _parse_tools_from_response(
        self,
        response: str,
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from the response.

        Args:
            response: The response string to extract tool calls from.

        Returns:
            List of parsed tool calls.
        """
        tool_calls = []

        # Pattern to match tool calls in JSON format
        tool_call_pattern = (
            r'\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}'
        )
        matches = re.findall(tool_call_pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue

        return tool_calls

    async def a_evaluate(
        self,
        query: Union[str, List[Dict[str, Any]]],
        tool_definitions: Union[Dict[str, Any], List[Dict[str, Any]]],
        tool_calls: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
        response: Union[str, List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> GraderScore:
        """Evaluate tool call accuracy. Accepts a query, tool definitions, and tool calls for evaluation.

        This method evaluates the accuracy of tool calls based on multiple criteria including relevance,
        parameter correctness, completeness, efficiency, and execution success. It assigns a score from
        1 to 5 based on how well the tool calls address the user's query.

        Args:
            query: Query or Chat history up to the message that has the tool call being evaluated.
                  Can be a string for simple queries or a list of message dictionaries for conversation history.
            tool_definitions: List of tool definitions whose calls are being evaluated.
                             Each definition includes name, description, and parameters information.
            tool_calls: Optional List of tool calls to evaluate. If not provided, response should be
                       provided and should contain tool call(s).
            response: Optional response to be evaluated alongside the tool calls.
                     If provided and tool_calls parameter is not provided, all tool calls in response will be evaluated.
                     If both response and tool_calls parameters are provided, only the tool calls in
                     tool_calls parameter will be evaluated.
            **kwargs: Additional keyword arguments passed to the underlying evaluation model.

        Returns:
            GraderScore with the evaluation result containing:
                - score: A numerical score between 1-5 indicating tool call accuracy
                - reason: Explanation of how the score was determined
                - metadata: Additional evaluation information

        Example:
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = ToolCallAccuracyGrader(model=model)
            >>> conversation = [
            ...     {"role": "user", "content": "What's the weather like in New York?"}
            ... ]
            >>> tool_defs = [
            ...     {
            ...         "name": "get_weather",
            ...         "description": "Get weather information for a location",
            ...         "parameters": {"location": "City name"}
            ...     }
            ... ]
            >>> tool_calls = [
            ...     {
            ...         "name": "get_weather",
            ...         "arguments": {"location": "New York"},
            ...         "result": {"temperature": 25, "condition": "sunny"}
            ...     }
            ... ]
            >>> result = asyncio.run(grader.a_evaluate(
            ...     query=conversation,
            ...     tool_definitions=tool_defs,
            ...     tool_calls=tool_calls
            ... ))
            >>> print(f"Score: {result.score}")
            Score: 5.0
        """
        # Handle tool calls extraction from response if needed
        if response and not tool_calls:
            parsed_tool_calls = self._parse_tools_from_response(str(response))
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls

        # Check if we have tool calls to evaluate
        if not tool_calls:
            return GraderScore(
                score=0.0,
                reason="No tool calls found in the response to evaluate",
                metadata={
                    "error": "No tool calls provided or found in response",
                },
            )

        # Ensure tool_calls and tool_definitions are lists
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tool_definitions, list):
            tool_definitions = [tool_definitions] if tool_definitions else []

        # Call parent evaluate method with the structured data
        result = await super().a_evaluate(
            query=json.dumps(query, indent=2),
            tool_calls=json.dumps(tool_calls, indent=2),
            tool_definitions=json.dumps(tool_definitions, indent=2),
            **kwargs,
        )

        # Process and normalize the result
        if hasattr(result, "score"):
            # Ensure score is within valid range
            score = max(1.0, min(5.0, result.score))
            result.score = score

        return result


if __name__ == "__main__":
    import asyncio

    async def main():
        # Initialize the grader
        model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
        grader = ToolCallAccuracyGrader(model=model)

        # Example conversation
        conversation = [
            {
                "role": "user",
                "content": "Can you check the weather in London and Paris?",
            },
        ]

        # Define the available tools
        tool_definitions = [
            {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        },
                    },
                    "required": ["location"],
                },
            },
        ]

        # Tool calls made by the agent
        tool_calls = [
            {
                "name": "get_weather",
                "arguments": {"location": "London"},
                "result": {"temperature": 18, "condition": "cloudy"},
            },
            {
                "name": "get_weather",
                "arguments": {"location": "Paris"},
                "result": {"temperature": 22, "condition": "sunny"},
            },
        ]

        # Evaluate the tool calls
        result = await grader.a_evaluate(
            query=conversation,
            tool_definitions=tool_definitions,
            tool_calls=tool_calls,
        )

        print("Tool Call Accuracy Evaluation Result:")
        print(f"Score: {result.score}")
        print(f"Reason: {result.reason}")
        if hasattr(result, "metadata") and result.metadata:
            print(f"Metadata: {result.metadata}")

    # Run the example
    asyncio.run(main())
