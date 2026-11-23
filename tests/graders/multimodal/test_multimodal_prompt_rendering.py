#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for multimodal prompt rendering.

Each multimodal grader declares its evaluation instructions in the user message
of its ``PromptTemplate``. These tests assert that the rendered request actually
carries those instructions (plus the image) instead of only the grader's JSON
format system prompt.
"""

from typing import Any, Dict, List

import pytest

from openjudge.graders.multimodal._internal import MLLMImage
from openjudge.graders.multimodal.image_coherence import ImageCoherenceGrader
from openjudge.graders.multimodal.image_helpfulness import ImageHelpfulnessGrader
from openjudge.graders.multimodal.text_to_image import TextToImageGrader
from openjudge.models.base_chat_model import BaseChatModel


class _Response:
    def __init__(self, parsed: Dict[str, Any]):
        self.parsed = parsed


class _CapturingModel(BaseChatModel):
    """Records every message list sent to the model."""

    def __init__(self, parsed: Dict[str, Any] | List[Dict[str, Any]]):
        super().__init__(model="capturing-model", stream=False)
        self.parsed = parsed
        self.calls: List[List[Dict[str, Any]]] = []

    async def achat(self, *args: Any, **kwargs: Any) -> _Response:
        self.calls.append(kwargs["messages"])
        response = self.parsed.pop(0) if isinstance(self.parsed, list) else self.parsed
        return _Response(response)


def _text_of(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(part["text"] for part in content if part.get("type") == "text")
    return "\n".join(parts)


def _image_urls_of(messages: List[Dict[str, Any]]) -> List[str]:
    urls: List[str] = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, str):
            urls.extend(part["image_url"]["url"] for part in content if part.get("type") == "image_url")
    return urls


IMAGE = MLLMImage(url="https://example.com/chart.png")


@pytest.mark.unit
class TestMultimodalPromptRendering:
    """The rendered request must contain the template's evaluation instructions."""

    @pytest.mark.asyncio
    async def test_image_coherence_sends_rubric_and_context(self):
        model = _CapturingModel({"score": 4.0, "reason": "ok"})
        grader = ImageCoherenceGrader(model=model)

        await grader.aevaluate(response=["Q3 sales rose 25%.", IMAGE, "Growth from new products."])

        sent = model.calls[0]
        text = _text_of(sent)
        assert "Q3 sales rose 25%." in text
        assert "Growth from new products." in text
        assert "Assess how coherent the image is" in text
        assert _image_urls_of(sent) == [IMAGE.url]

    @pytest.mark.asyncio
    async def test_image_coherence_custom_rubric_string_replaces_default(self):
        """A custom rubric string replaces the built-in generic rubric"""
        model = _CapturingModel({"score": 4, "reason": "Synthetic offline response"})
        grader = ImageCoherenceGrader(
            model=model,
            rubrics="Chart values must match the paragraph numbers.",
        )
        await grader.aevaluate(response=["Q3 sales rose 25%.", IMAGE, "Growth from new products."])
        text = _text_of(model.calls[0])
        assert "Chart values must match the paragraph numbers." in text
        assert "Assess how coherent the image is" not in text

    @pytest.mark.asyncio
    async def test_image_coherence_rubric_list_renders_as_bullets(self):
        """A rubric list renders as bullet lines"""
        model = _CapturingModel({"score": 4, "reason": "Synthetic offline response"})
        grader = ImageCoherenceGrader(
            model=model,
            rubrics=["Numbers match", "Placement is logical"],
        )
        await grader.aevaluate(response=["Q3 sales rose 25%.", IMAGE, "Growth from new products."])
        text = _text_of(model.calls[0])
        assert "- Numbers match" in text
        assert "- Placement is logical" in text

    @pytest.mark.asyncio
    async def test_image_coherence_default_rubric_unchanged(self):
        """Without rubrics the built-in rubric is byte-identical to history"""
        model = _CapturingModel({"score": 4, "reason": "Synthetic offline response"})
        grader = ImageCoherenceGrader(model=model)
        await grader.aevaluate(response=["Q3 sales rose 25%.", IMAGE, "Growth from new products."])
        text = _text_of(model.calls[0])
        assert (
            "Assess how coherent the image is in relation to its accompanying text, "
            "assigning a score from 1 to 5." in text
        )

    @pytest.mark.asyncio
    async def test_image_helpfulness_sends_rubric_and_context(self):
        model = _CapturingModel({"score": 4.0, "reason": "ok"})
        grader = ImageHelpfulnessGrader(model=model)

        await grader.aevaluate(response=["The system architecture:", IMAGE, "It shows the layers."])

        sent = model.calls[0]
        text = _text_of(sent)
        assert "The system architecture:" in text
        assert "It shows the layers." in text
        assert "<Steps>" in text
        assert _image_urls_of(sent) == [IMAGE.url]

    @pytest.mark.asyncio
    async def test_text_to_image_sends_query_and_rubrics(self):
        model = _CapturingModel(
            [
                {"score": 4.0, "reason": "ok"},
                {"score": [4.0, 3.0], "reason": "ok"},
            ],
        )
        grader = TextToImageGrader(model=model)

        await grader.aevaluate(query="A cat sitting on a blue sofa", response=IMAGE)

        semantic, perceptual = model.calls

        semantic_text = _text_of(semantic)
        assert "A cat sitting on a blue sofa" in semantic_text
        assert "Evaluate how successfully the AI-generated image follows the text prompt" in semantic_text
        assert _image_urls_of(semantic) == [IMAGE.url]

        perceptual_text = _text_of(perceptual)
        assert "Evaluate the perceptual quality" in perceptual_text
        assert "Naturalness (first score)" in perceptual_text
        assert _image_urls_of(perceptual) == [IMAGE.url]
