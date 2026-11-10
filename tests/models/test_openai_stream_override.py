# -*- coding: utf-8 -*-
"""Regression coverage for per-call OpenAI streaming options."""

from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from openjudge.models import OpenAIChatModel
from openjudge.models.schema.oai.response import ChatResponse


class PersonForStreamTesting(BaseModel):
    """Structured result for both response modes."""

    name: str
    age: int


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("instance_stream,call_stream", [(False, None), (True, None), (False, True), (True, False)])
@pytest.mark.parametrize("structured", [False, True])
async def test_call_stream_option_controls_response_mode(
    instance_stream: bool,
    call_stream: bool | None,
    structured: bool,
) -> None:
    """Return usable content using the call override or constructor default."""
    content = '{"name":"Ada","age":37}'
    effective_stream = instance_stream if call_stream is None else call_stream
    completion = ChatCompletion(
        id="test",
        object="chat.completion",
        created=1,
        model="qwen-test",
        choices=[{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    )

    async def chunks() -> AsyncGenerator[ChatCompletionChunk, None]:
        yield ChatCompletionChunk(
            id="test",
            object="chat.completion.chunk",
            created=1,
            model="qwen-test",
            choices=[{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        )

    with patch("openjudge.models.openai_chat_model.AsyncOpenAI") as factory:
        client = factory.return_value
        client.chat.completions.create = AsyncMock(return_value=chunks() if effective_stream else completion)
        client.chat.completions.parse = AsyncMock(return_value=completion)
        model = OpenAIChatModel(model="qwen-test", api_key="test-key", stream=instance_stream)

        def callback(_response: ChatResponse) -> dict[str, bool]:
            return {"callback": True}

        response = await model.achat(
            messages=[{"role": "user", "content": "Return JSON for Ada"}],
            structured_model=PersonForStreamTesting if structured else None,
            callback=callback,
            **({} if call_stream is None else {"stream": call_stream}),
        )
        if effective_stream:
            responses = [item async for item in response]
            assert responses
            response = responses[-1]
        assert response.content == content
        expected = {"name": "Ada", "age": 37} if structured else {}
        assert response.parsed == {**expected, "callback": True}
        assert model.stream is instance_stream

        if structured and not effective_stream:
            client.chat.completions.parse.assert_awaited_once()
            client.chat.completions.create.assert_not_awaited()
            outgoing = client.chat.completions.parse.call_args.kwargs
            assert "stream" not in outgoing
        else:
            client.chat.completions.create.assert_awaited_once()
            client.chat.completions.parse.assert_not_awaited()
            outgoing = client.chat.completions.create.call_args.kwargs
            assert outgoing["stream"] is effective_stream
        if effective_stream:
            assert "extra_body" not in outgoing
        else:
            assert outgoing["extra_body"]["enable_thinking"] is False
