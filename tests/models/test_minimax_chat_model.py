#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and integration tests for MiniMaxChatModel.

Unit tests run offline using mocks.
Integration tests require the MINIMAX_API_KEY environment variable.

Run unit tests only:
    pytest tests/models/test_minimax_chat_model.py -m unit -v

Run all tests (requires API key):
    pytest tests/models/test_minimax_chat_model.py -v
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.models import MiniMaxChatModel
from openjudge.models.minimax_chat_model import MINIMAX_MODELS, _strip_think_tags
from openjudge.models.schema.oai.response import ChatResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_response(content: str) -> ChatResponse:
    return ChatResponse(role="assistant", content=content)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMiniMaxChatModelInit:
    """Verify constructor defaults and argument handling."""

    def test_default_base_url(self):
        model = MiniMaxChatModel(api_key="test-key")
        assert model.client.base_url is not None
        assert "minimax.io" in str(model.client.base_url)

    def test_default_model(self):
        model = MiniMaxChatModel(api_key="test-key")
        assert model.model == "MiniMax-M2.7"

    def test_custom_model(self):
        model = MiniMaxChatModel(model="MiniMax-M2.5", api_key="test-key")
        assert model.model == "MiniMax-M2.5"

    def test_all_supported_models(self):
        for name in MINIMAX_MODELS:
            m = MiniMaxChatModel(model=name, api_key="test-key")
            assert m.model == name

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key-123")
        model = MiniMaxChatModel()
        # The key is passed into the AsyncOpenAI client — check kwargs stored
        assert model.client.api_key == "env-key-123"

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        model = MiniMaxChatModel(api_key="explicit-key")
        assert model.client.api_key == "explicit-key"

    def test_temperature_clamped_above_one(self):
        model = MiniMaxChatModel(api_key="test-key", temperature=1.5)
        assert model.kwargs["temperature"] == pytest.approx(1.0)

    def test_temperature_clamped_zero(self):
        model = MiniMaxChatModel(api_key="test-key", temperature=0.0)
        assert model.kwargs["temperature"] > 0.0

    def test_temperature_valid_unchanged(self):
        model = MiniMaxChatModel(api_key="test-key", temperature=0.7)
        assert model.kwargs["temperature"] == pytest.approx(0.7)

    def test_custom_base_url(self):
        model = MiniMaxChatModel(api_key="k", base_url="https://custom.minimax.io/v1")
        assert "custom.minimax.io" in str(model.client.base_url)


@pytest.mark.unit
class TestStripThinkTags:
    """Test the think-tag stripping utility."""

    def test_strips_single_think_block(self):
        text = "<think>internal reasoning</think>Final answer."
        assert _strip_think_tags(text) == "Final answer."

    def test_strips_multiline_think_block(self):
        text = "<think>\nline1\nline2\n</think>Result"
        assert _strip_think_tags(text) == "Result"

    def test_no_think_tags_unchanged(self):
        text = "No thinking tags here."
        assert _strip_think_tags(text) == text

    def test_multiple_think_blocks(self):
        text = "<think>step1</think>middle<think>step2</think>end"
        assert _strip_think_tags(text) == "middleend"

    def test_empty_think_tag(self):
        text = "<think></think>answer"
        assert _strip_think_tags(text) == "answer"


@pytest.mark.unit
class TestMiniMaxChatModelAchat:
    """Verify achat() behavior using mocked super().achat()."""

    @pytest.mark.asyncio
    async def test_returns_clean_response(self):
        model = MiniMaxChatModel(api_key="test-key")
        raw = _make_chat_response("<think>thinking</think>Clean answer.")

        with patch.object(
            MiniMaxChatModel.__bases__[0],
            "achat",
            new_callable=AsyncMock,
            return_value=raw,
        ):
            result = await model.achat(messages=[{"role": "user", "content": "Hi"}])

        assert isinstance(result, ChatResponse)
        assert result.content == "Clean answer."

    @pytest.mark.asyncio
    async def test_no_think_tags_unchanged(self):
        model = MiniMaxChatModel(api_key="test-key")
        raw = _make_chat_response("Direct response without reasoning.")

        with patch.object(
            MiniMaxChatModel.__bases__[0],
            "achat",
            new_callable=AsyncMock,
            return_value=raw,
        ):
            result = await model.achat(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Direct response without reasoning."

    @pytest.mark.asyncio
    async def test_temperature_clamped_in_kwargs(self):
        model = MiniMaxChatModel(api_key="test-key")
        raw = _make_chat_response("ok")

        captured_kwargs = {}

        async def fake_achat(self_inner, messages, **kwargs):
            captured_kwargs.update(kwargs)
            return raw

        with patch.object(MiniMaxChatModel.__bases__[0], "achat", fake_achat):
            await model.achat(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=2.5,
            )

        assert captured_kwargs["temperature"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_temperature_zero_clamped(self):
        model = MiniMaxChatModel(api_key="test-key")
        raw = _make_chat_response("ok")

        captured_kwargs = {}

        async def fake_achat(self_inner, messages, **kwargs):
            captured_kwargs.update(kwargs)
            return raw

        with patch.object(MiniMaxChatModel.__bases__[0], "achat", fake_achat):
            await model.achat(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.0,
            )

        assert captured_kwargs["temperature"] > 0.0

    @pytest.mark.asyncio
    async def test_streaming_response_not_modified(self):
        """For streaming generators the response is returned as-is."""
        model = MiniMaxChatModel(api_key="test-key", stream=True)

        async def fake_generator():
            yield _make_chat_response("chunk1")

        generator = fake_generator()

        with patch.object(
            MiniMaxChatModel.__bases__[0],
            "achat",
            new_callable=AsyncMock,
            return_value=generator,
        ):
            result = await model.achat(messages=[{"role": "user", "content": "Hi"}])

        # Streaming result should be the generator, not a ChatResponse
        assert hasattr(result, "__aiter__")


@pytest.mark.unit
class TestMiniMaxChatModelInheritance:
    """Verify class hierarchy and exported symbols."""

    def test_inherits_from_openai_chat_model(self):
        from openjudge.models.openai_chat_model import OpenAIChatModel

        assert issubclass(MiniMaxChatModel, OpenAIChatModel)

    def test_exported_from_models_package(self):
        from openjudge.models import MiniMaxChatModel as Imported

        assert Imported is MiniMaxChatModel

    def test_minimax_models_list_not_empty(self):
        assert len(MINIMAX_MODELS) >= 4

    def test_minimax_models_contain_m27(self):
        assert "MiniMax-M2.7" in MINIMAX_MODELS

    def test_minimax_models_contain_highspeed(self):
        assert any("highspeed" in m for m in MINIMAX_MODELS)


# ---------------------------------------------------------------------------
# Integration tests (require MINIMAX_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
class TestMiniMaxChatModelIntegration:
    """Live API tests — require MINIMAX_API_KEY."""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        model = MiniMaxChatModel(model="MiniMax-M2.7")
        response = await model.achat(messages=[{"role": "user", "content": "Reply with the single word: hello"}])
        assert isinstance(response, ChatResponse)
        assert response.content
        # Think-tags should be stripped
        assert "<think>" not in response.content

    @pytest.mark.asyncio
    async def test_temperature_clamping_does_not_error(self):
        model = MiniMaxChatModel(model="MiniMax-M2.7", temperature=0.0)
        response = await model.achat(messages=[{"role": "user", "content": "Say: ok"}])
        assert isinstance(response, ChatResponse)

    @pytest.mark.asyncio
    async def test_highspeed_model(self):
        model = MiniMaxChatModel(model="MiniMax-M2.7-highspeed")
        response = await model.achat(messages=[{"role": "user", "content": "Reply with the single word: hello"}])
        assert isinstance(response, ChatResponse)
        assert response.content
