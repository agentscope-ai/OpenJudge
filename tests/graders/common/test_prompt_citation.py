#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests verifying that common grader prompts include the citation instruction
("citing specific text from the response" / "引用回复中的具体文本")
in both English and Chinese output schemas.
"""

from unittest.mock import AsyncMock

import pytest

from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.hallucination import HallucinationGrader
from openjudge.graders.common.harmfulness import HarmfulnessGrader
from openjudge.graders.common.instruction_following import InstructionFollowingGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.models.schema.prompt_template import LanguageEnum

EN_CITE_PHRASE = "citing specific text from the response"
ZH_CITE_PHRASE = "引用回复中的具体文本"

COMMON_GRADERS = [
    CorrectnessGrader,
    HallucinationGrader,
    HarmfulnessGrader,
    InstructionFollowingGrader,
    RelevanceGrader,
]


def _get_prompt_content(grader_cls, language):
    """Instantiate grader and return combined prompt text for the given language."""
    grader = grader_cls(model=AsyncMock())
    template = grader.get_template(language=language)
    messages = template[language.value]
    return " ".join(msg["content"] for msg in messages)


@pytest.mark.unit
class TestPromptCitationInstruction:
    """Verify that all common grader prompts require citing specific text."""

    @pytest.mark.parametrize("grader_cls", COMMON_GRADERS, ids=lambda c: c.__name__)
    def test_en_prompt_contains_citation_instruction(self, grader_cls):
        combined = _get_prompt_content(grader_cls, LanguageEnum.EN)
        assert EN_CITE_PHRASE in combined, f"{grader_cls.__name__} EN prompt missing citation instruction"

    @pytest.mark.parametrize("grader_cls", COMMON_GRADERS, ids=lambda c: c.__name__)
    def test_zh_prompt_contains_citation_instruction(self, grader_cls):
        combined = _get_prompt_content(grader_cls, LanguageEnum.ZH)
        assert ZH_CITE_PHRASE in combined, f"{grader_cls.__name__} ZH prompt missing citation instruction"
