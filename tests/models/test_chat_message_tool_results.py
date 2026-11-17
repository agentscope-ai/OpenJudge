# -*- coding: utf-8 -*-
"""Typed and dictionary tool results must reach the model with the same ID."""

import json

import httpx
import pytest

from openjudge.models import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("content", ["4", ""])
async def test_tool_results_reach_the_real_openai_client(typed: bool, content: str) -> None:
    """Continue a tool dialogue through the public model and actual HTTP serializer."""
    messages = [
        {"role": "user", "content": "What is two plus two?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_sum", "type": "function", "function": {"name": "sum", "arguments": '{"a":2,"b":2}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_sum", "content": content},
    ]
    inputs = [ChatMessage.from_dict(message) for message in messages] if typed else messages
    requests = []

    def complete(request: httpx.Request) -> httpx.Response:
        """Supply only the external completion; retain OpenAI's real request encoding."""
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion-local",
                "object": "chat.completion",
                "created": 1,
                "model": "local-test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "The answer is four."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(complete)) as client:
        model = OpenAIChatModel(
            model="local-test-model",
            api_key="local-test-key",
            base_url="https://model.example.invalid/v1",
            client_args={"http_client": client},
        )
        result = await model.achat(inputs)

    assert result.content == "The answer is four."
    assert len(requests) == 1
    assert requests[0]["messages"] == messages


@pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
def test_ordinary_messages_do_not_gain_a_tool_call_id(role: str) -> None:
    """Keep serialization of existing non-tool message roles unchanged."""
    data = {"role": role, "content": "Ordinary content"}
    assert ChatMessage.from_dict(data).to_dict() == data


@pytest.mark.asyncio
async def test_tool_messages_still_require_a_call_id_before_sending() -> None:
    """The existing public model guard also applies to typed tool messages."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(500))) as client:
        model = OpenAIChatModel(
            model="local-test-model",
            api_key="local-test-key",
            base_url="https://model.example.invalid/v1",
            client_args={"http_client": client},
        )
        with pytest.raises(ValueError, match="Tool messages need 'tool_call_id'"):
            await model.achat([ChatMessage(role="tool", content="4")])
