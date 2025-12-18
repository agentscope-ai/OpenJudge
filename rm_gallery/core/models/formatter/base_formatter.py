# -*- coding: utf-8 -*-
"""The formatter module."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from rm_gallery.core.models.schema.oai.message import ChatMessage


class BaseFormatter(ABC):
    """The base class for formatters that convert messages between different formats.

    This class defines the interface for converting messages from external formats
    to OpenAI format and vice versa.
    """

    @abstractmethod
    def format_to_openai(self, messages: List[Dict[str, Any]]) -> List[ChatMessage]:
        """Convert messages from the source format to OpenAI format.

        Args:
            messages: List of dictionaries in the source format.

        Returns:
            List of ChatMessage objects in OpenAI format.
        """

    @abstractmethod
    def format_from_openai(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert messages from OpenAI format to the source format.

        Args:
            messages: List of ChatMessage objects in OpenAI format.

        Returns:
            List of dictionaries in the source format.
        """
