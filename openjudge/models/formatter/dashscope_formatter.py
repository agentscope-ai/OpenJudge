# -*- coding: utf-8 -*-
"""Formatter for DashScope models."""

from typing import Any, Dict, List, Union

from openjudge.models.formatter.base_formatter import BaseFormatter
from openjudge.models.schema.oai.message import ChatMessage


class DashScopeFormatter(BaseFormatter):
    """Formatter for converting between DashScope and OpenAI message formats."""

    def format_to_openai(self, messages: List[Dict[str, Any]]) -> List[ChatMessage]:
        """Convert DashScope format messages to OpenAI format.

        Args:
            messages: List of dictionaries in DashScope format.

        Returns:
            List of ChatMessage objects in OpenAI format.
        """
        openai_messages = []

        for msg in messages:
            # Convert DashScope message format to OpenAI format
            openai_msg = {
                "role": msg.get("role", "user"),
                "content": self._convert_content_to_openai(msg.get("content", "")),
            }

            # Add name if present
            if "name" in msg:
                openai_msg["name"] = msg["name"]

            # Create ChatMessage object
            chat_message = ChatMessage(**openai_msg)
            openai_messages.append(chat_message)

        return openai_messages

    def format_from_openai(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert OpenAI format messages to DashScope format.

        Args:
            messages: List of ChatMessage objects in OpenAI format.

        Returns:
            List of dictionaries in DashScope format.
        """
        dashscope_messages = []

        for msg in messages:
            # Convert ChatMessage to DashScope format
            dashscope_msg = {
                "role": msg.role,
                "content": self._convert_content_to_dashscope(msg.content),
            }

            # Add name if present
            if msg.name:
                dashscope_msg["name"] = msg.name

            dashscope_messages.append(dashscope_msg)

        return dashscope_messages

    def _convert_content_to_openai(
        self, content: Union[str, List[Dict[str, Any]], None]
    ) -> Union[str, List[Dict[str, Any]]]:
        """Convert content from DashScope format to OpenAI format.

        Args:
            content: Content in DashScope format (string or list of content parts).

        Returns:
            Content in OpenAI format.
        """
        # Return empty string if no content input
        if not content:
            return ""

        # If content is a string, return as is
        if isinstance(content, str):
            return content

        # If content is a list, process each part
        if isinstance(content, list):
            openai_content = []
            for part in content:
                if isinstance(part, dict):
                    # Handle different types of content parts
                    part_type = part.get("type")

                    if part_type == "text":
                        openai_content.append({"type": "text", "text": part.get("text", "")})

                    elif part_type == "image":
                        # Convert image to OpenAI format
                        image_data = part.get("image", "")
                        openai_content.append({"type": "image_url", "image_url": {"url": image_data}})

                    # Handle audio content
                    elif part_type == "audio":
                        # Convert audio to OpenAI format
                        audio_data = part.get("audio", "")
                        openai_content.append({"type": "input_audio", "input_audio": {"data": audio_data}})

                    # Handle video content
                    elif part_type == "video":
                        # Convert video to OpenAI format (as video_url)
                        video_data = part.get("video", "")
                        openai_content.append({"type": "video_url", "video_url": {"url": video_data}})

                    # For tool calls and functions, keep as is
                    elif part_type in ["tool_call", "function"]:
                        openai_content.append(part)

                    # For any other type, keep as is
                    else:
                        openai_content.append(part)

            return openai_content

        # For any other type, convert to string
        return str(content)

    def _convert_content_to_dashscope(
        self, content: Union[str, List[Dict[str, Any]], None]
    ) -> Union[str, List[Dict[str, Any]]]:
        """Convert content from OpenAI format to DashScope format.

        Args:
            content: Content in OpenAI format (string or list of content parts).

        Returns:
            Content in DashScope format.
        """
        # Return empty string if no content input
        if not content:
            return ""

        # If content is a string, return as is
        if isinstance(content, str):
            return content

        # If content is a list, process each part
        if isinstance(content, list):
            dashscope_content = []
            for part in content:
                if isinstance(part, dict):
                    # Handle different types of content parts
                    part_type = part.get("type")

                    if part_type == "text":
                        dashscope_content.append({"type": "text", "text": part.get("text", "")})

                    elif part_type == "image_url":
                        # Convert image_url to DashScope format
                        image_url = part.get("image_url", {}).get("url", "")
                        dashscope_content.append({"type": "image", "image": image_url})

                    # Handle audio content
                    elif part_type == "input_audio":
                        # Convert input_audio to DashScope format
                        audio_data = part.get("input_audio", {})
                        audio_url = audio_data.get("data", "")
                        dashscope_content.append({"type": "audio", "audio": audio_url})

                    # Handle tool calls
                    elif part_type == "tool_call":
                        # Keep tool_call as is for DashScope compatibility
                        dashscope_content.append(part)

                    # Handle function calls
                    elif part_type == "function":
                        # Keep function as is for DashScope compatibility
                        dashscope_content.append(part)

                    # Handle video content (if supported)
                    elif part_type == "video_url":
                        # Convert video_url to DashScope format
                        video_url = part.get("video_url", {}).get("url", "")
                        dashscope_content.append({"type": "video", "video": video_url})

                    # For any other type, keep as is
                    else:
                        dashscope_content.append(part)

            return dashscope_content

        # For any other type, convert to string
        return str(content)
