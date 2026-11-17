# -*- coding: utf-8 -*-
"""Automatic Qwen defaults must stay local to the request that needs them."""

import json
from typing import Any

import httpx
import pytest

from openjudge.models import OpenAIChatModel

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def endpoint(requests: list[dict[str, Any]]) -> httpx.MockTransport:
    """Supply JSON and SSE completions behind the actual OpenAI HTTP client."""

    def complete(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        common = {"id": "local", "created": 1, "model": payload["model"]}
        if payload["stream"]:
            chunks = [
                {
                    **common,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": "ready"}, "finish_reason": None}
                    ],
                },
                {
                    **common,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                **common,
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}
                ],
            },
        )

    return httpx.MockTransport(complete)


@pytest.mark.parametrize(
    "model_name,scope",
    [("qwen-local-test", "constructor"), ("pai-judge-local", "constructor"), ("qwen-local-test", "call")],
)
async def test_nonstream_default_does_not_reach_a_later_stream(model_name: str, scope: str) -> None:
    """A default for one request must not alter reusable caller options."""
    body = {"seed": 7}
    requests: list[dict[str, Any]] = []
    options = {"extra_body": body}
    async with httpx.AsyncClient(transport=endpoint(requests)) as client:
        model = OpenAIChatModel(
            model=model_name,
            api_key="local-test-key",
            base_url="https://model.example.invalid/v1",
            client_args={"http_client": client},
            **(options if scope == "constructor" else {}),
        )
        call_options = options if scope == "call" else {}
        first = await model.achat([{"role": "user", "content": "ready?"}], **call_options)
        stream = await model.achat([{"role": "user", "content": "ready?"}], stream=True, **call_options)
        chunks = [chunk async for chunk in stream]
    assert first.content == chunks[-1].content == "ready"
    assert requests[0]["enable_thinking"] is False
    assert "enable_thinking" not in requests[1]
    assert [request["seed"] for request in requests] == [7, 7]
    assert body == {"seed": 7}


@pytest.mark.parametrize("scope", ["constructor", "call"])
async def test_none_extra_body_still_gets_nonstream_default(scope: str) -> None:
    """An optional SDK body may be None instead of an empty mapping."""
    requests: list[dict[str, Any]] = []
    options = {"extra_body": None}
    async with httpx.AsyncClient(transport=endpoint(requests)) as client:
        model = OpenAIChatModel(
            model="qwen-local-test",
            api_key="local-test-key",
            base_url="https://model.example.invalid/v1",
            client_args={"http_client": client},
            **(options if scope == "constructor" else {}),
        )
        result = await model.achat([{"role": "user", "content": "ready?"}], **(options if scope == "call" else {}))
    assert result.content == "ready"
    assert requests[0]["enable_thinking"] is False
    assert options == {"extra_body": None}


@pytest.mark.parametrize("scope", ["constructor", "call"])
@pytest.mark.parametrize("enabled", [True, False])
async def test_explicit_thinking_is_preserved(scope: str, enabled: bool) -> None:
    """Caller-selected thinking takes precedence over the automatic default."""
    requests: list[dict[str, Any]] = []
    body = {"enable_thinking": enabled, "seed": 7}
    options = {"extra_body": body}
    async with httpx.AsyncClient(transport=endpoint(requests)) as client:
        model = OpenAIChatModel(
            model="qwen-local-test",
            api_key="local-test-key",
            base_url="https://model.example.invalid/v1",
            client_args={"http_client": client},
            **(options if scope == "constructor" else {}),
        )
        result = await model.achat([{"role": "user", "content": "ready?"}], **(options if scope == "call" else {}))
    assert result.content == "ready"
    assert requests[0]["enable_thinking"] is enabled
    assert requests[0]["seed"] == 7
    assert body == {"enable_thinking": enabled, "seed": 7}


@pytest.mark.parametrize("model_name", ["qwen-local-test", "ordinary-model"])
async def test_omitted_body_and_other_models(model_name: str) -> None:
    """The default remains limited to the supported nonstream model families."""
    requests: list[dict[str, Any]] = []
    async with httpx.AsyncClient(transport=endpoint(requests)) as client:
        model = OpenAIChatModel(
            model=model_name,
            api_key="local-test-key",
            base_url="https://model.example.invalid/v1",
            client_args={"http_client": client},
        )
        await model.achat([{"role": "user", "content": "ready?"}])
        stream = await model.achat([{"role": "user", "content": "ready?"}], stream=True)
        chunks = [chunk async for chunk in stream]
    assert chunks[-1].content == "ready"
    assert ("enable_thinking" in requests[0]) == (model_name == "qwen-local-test")
    assert "enable_thinking" not in requests[1]
