# -*- coding: utf-8 -*-
"""Schema definition for chat model responses.

This module defines the data model for chat responses from language models,
following the OpenAI response format.
"""

from typing import Any, Dict, Optional

from pydantic import Field

from rm_gallery.core.models.schema.oai.message import ChatMessage


class ChatResponse(ChatMessage):
    """A complete chat response compatible with OpenAI's ChatCompletionMessage format.

    This class represents a complete chat completion message response from the OpenAI API,
    including all standard fields like id, choices, created, model, usage, etc.
    """

    parsed: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parsed response content")
