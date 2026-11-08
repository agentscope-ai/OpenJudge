# -*- coding: utf-8 -*-
"""
Prompt Rendering Utilities

Multimodal graders declare their evaluation instructions in the user message of
a ``PromptTemplate``. Chat models need those instructions and the image to
arrive together, so the rendered user message is turned into an OpenAI content
part list with the images appended.
"""

from typing import Any, Dict, List, Union

from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate
from openjudge.models.schema.qwen.mllmImage import MLLMImage, format_image_content


def build_image_messages(
    template: PromptTemplate,
    language: Union[LanguageEnum, str],
    images: List[MLLMImage],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Render a template and attach images to its user message

    Args:
        template: Template whose messages describe the evaluation task
        language: Language to render the template in
        images: Images to send alongside the instructions
        **kwargs: Values substituted into the template placeholders

    Returns:
        List of OpenAI-format message dicts, the last user message carrying both
        the rendered instructions and the images

    Raises:
        ValueError: If the template has no user message to attach images to

    Example:
        >>> messages = build_image_messages(
        ...     DEFAULT_IMAGE_COHERENCE_TEMPLATE,
        ...     LanguageEnum.EN,
        ...     [MLLMImage(url="https://example.com/chart.png")],
        ...     context_above="Sales data for Q3:",
        ...     context_below="Shows 20% growth",
        ... )
        >>> messages[-1]["content"][1]["type"]
        'image_url'
    """
    messages = template.format(language=language, **kwargs)

    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] != "user":
            continue
        text = messages[index]["content"]
        image_parts = format_image_content("", images)[1:]
        if isinstance(text, str):
            messages[index]["content"] = format_image_content(text, images)
        else:
            messages[index]["content"] = [*text, *image_parts]
        return messages

    raise ValueError("Template has no user message to attach the images to")


__all__ = ["build_image_messages"]
