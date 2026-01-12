# -*- coding: utf-8 -*-
"""
Qwen VLM API Integration

Provides integration with Alibaba Cloud's Qwen VL models via DashScope API.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Type, Union

from dashscope import MultiModalConversation
from loguru import logger
from pydantic import BaseModel

from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.response import ChatResponse
from openjudge.models.schema.qwen.mllmImage import MLLMImage


class QwenVLModel(BaseChatModel):
    """
    Qwen VL API client

    Provides a unified interface for calling Qwen VL models through DashScope.
    Supports both synchronous and asynchronous calls, structured outputs, and cost tracking.

    Example:
        >>> from openjudge.models import QwenVLModel
        >>> from openjudge.models.schema.qwen.mllmImage import MLLMImage
        >>>
        >>> # Initialize
        >>> model = QwenVLModel(
        ...     api_key=os.getenv("DASHSCOPE_API_KEY"),
        ...     model="qwen-vl-plus"
        ... )
        >>>
        >>> # Generate response
        >>> response = model.generate(
        ...     text="Describe this image",
        ...     images=[MLLMImage(url="https://example.com/image.jpg")]
        ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-plus",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2000,
    ):
        """
        Initialize Qwen VL API client

        Args:
            api_key: DashScope API key (defaults to DASHSCOPE_API_KEY env var)
            model: Model name
            temperature: Sampling temperature
            top_p: Nucleus sampling
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model=model, stream=False)

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set via DASHSCOPE_API_KEY environment variable",
            )

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Cost tracking
        self._total_requests = 0
        self._total_cost = 0.0

    def _format_messages(
        self,
        content: List[Union[str, "MLLMImage"]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format messages for Qwen VL API

        Args:
            content: List of text and images
            system_prompt: Optional system prompt

        Returns:
            Formatted messages list
        """
        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"text": system_prompt}]},
            )

        # Format user message with text and images
        user_content = []
        for item in content:
            if isinstance(item, str):
                user_content.append({"text": item})
            elif isinstance(item, MLLMImage):
                if item.url:
                    user_content.append({"image": item.url})
                elif item.base64:
                    # DashScope supports base64 images
                    user_content.append(
                        {
                            "image": f"data:image/{item.format or 'jpeg'};base64,{item.base64}",
                        },
                    )

        messages.append({"role": "user", "content": user_content})

        return messages

    def generate(
        self,
        text: str,
        images: Optional[List["MLLMImage"]] = None,
        schema: Optional[Type[BaseModel]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
    ) -> Union[str, BaseModel, Dict[str, Any]]:
        """
        Generate response synchronously

        Args:
            text: Text prompt
            images: Optional list of images
            schema: Optional Pydantic schema for structured output (deprecated, use response_format)
            response_format: Optional Pydantic schema for structured output
            system_prompt: Optional system prompt

        Returns:
            If response_format/schema provided: structured output (dict or BaseModel)
            Otherwise: response_text
        """
        # Build content list
        content: List[Union[str, MLLMImage]] = [text]
        if images:
            content.extend(images)

        # Format messages
        messages = self._format_messages(content, system_prompt)

        # Call API
        try:
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_length=self.max_tokens,
            )

            self._total_requests += 1

            if response.status_code != 200:
                raise RuntimeError(
                    f"API call failed with status {response.status_code}: " f"{response.message}",
                )

            # Extract text response
            response_text = response.output.choices[0].message.content[0]["text"]

            # Calculate cost (approximate)
            cost = self._estimate_cost(response)
            self._total_cost += cost

            # Parse structured output if schema or response_format provided
            output_schema = response_format or schema
            if output_schema:
                from openjudge.utils.utils import trim_and_load_json

                data = trim_and_load_json(response_text)
                try:
                    structured_output = output_schema(**data)
                    return structured_output
                except Exception as e:
                    logger.warning(
                        f"Failed to parse as Pydantic model: {e}, returning dict",
                    )
                    return data

            return response_text

        except Exception as e:
            logger.error(f"Qwen VL API call failed: {e}")
            raise

    async def a_generate(
        self,
        text: str,
        images: Optional[List["MLLMImage"]] = None,
        schema: Optional[Type[BaseModel]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
    ) -> Union[str, BaseModel, Dict[str, Any]]:
        """
        Generate response asynchronously

        Args:
            text: Text prompt
            images: Optional list of images
            schema: Optional Pydantic schema for structured output (deprecated)
            response_format: Optional Pydantic schema for structured output
            system_prompt: Optional system prompt

        Returns:
            If response_format/schema provided: structured output
            Otherwise: response_text
        """

        # DashScope doesn't have native async support yet
        # Run in executor to avoid blocking
        return await asyncio.to_thread(
            self.generate,
            text,
            images,
            schema,
            response_format,
            system_prompt,
        )

    async def achat(
        self,
        text: str,
        images: List[MLLMImage] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Make the API callable

        This implements the BaseChatModel interface.
        """

        # Generate response
        response_text = await self.a_generate(text, images)
        # Return ChatResponse
        return ChatResponse(
            content=response_text,
        )

    def _estimate_cost(self, response: Any) -> float:
        """
        Estimate API call cost

        Args:
            response: API response object

        Returns:
            Estimated cost in USD
        """
        # Rough estimates for Qwen VL models (as of 2024)
        # qwen-vl-plus: ~$0.001 per request
        # qwen-vl-max: ~$0.002 per request
        del response
        if "max" in self.model.lower():
            return 0.002
        else:
            return 0.001

    def generate_from_parts(
        self,
        parts: List[Union[str, MLLMImage]],
        response_format: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
    ) -> Union[str, BaseModel, Dict[str, Any]]:
        """
        Generate response from a list of text and image parts

        Args:
            parts: List of text strings and MLLMImage objects
            response_format: Optional Pydantic schema for structured output
            system_prompt: Optional system prompt

        Returns:
            Generated response (text or structured)
        """
        # Separate text and images
        text_parts = []
        image_parts = []

        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, MLLMImage):
                image_parts.append(part)

        # Combine text parts
        combined_text = "\n".join(text_parts)

        # Generate
        return self.generate(
            text=combined_text,
            images=image_parts if image_parts else None,
            response_format=response_format,
            system_prompt=system_prompt,
        )

    async def a_generate_from_parts(
        self,
        parts: List[Union[str, MLLMImage]],
        response_format: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
    ) -> Union[str, BaseModel, Dict[str, Any]]:
        """
        Generate response from parts asynchronously

        Args:
            parts: List of text strings and MLLMImage objects
            response_format: Optional Pydantic schema for structured output
            system_prompt: Optional system prompt

        Returns:
            Generated response (text or structured)
        """
        # Separate text and images
        text_parts = []
        image_parts = []

        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, MLLMImage):
                image_parts.append(part)

        # Combine text parts
        combined_text = "\n".join(text_parts)

        # Generate
        return await self.a_generate(
            text=combined_text,
            images=image_parts if image_parts else None,
            response_format=response_format,
            system_prompt=system_prompt,
        )

    def get_cost_stats(self) -> Dict[str, Any]:
        """
        Get cost statistics

        Returns:
            Dictionary with cost statistics
        """
        return {
            "total_requests": self._total_requests,
            "total_cost_usd": self._total_cost,
            "average_cost_per_request": (self._total_cost / self._total_requests if self._total_requests > 0 else 0.0),
            "model": self.model,
        }
