# -*- coding: utf-8 -*-
"""
Base configuration classes for local model providers.

This module defines the base classes and enums for local model provider
configurations like Ollama.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class ProviderType(str, Enum):
    """Enum for supported local model providers.

    Attributes:
        OLLAMA: Ollama local inference server.
    """

    OLLAMA = "ollama"


class BaseProviderConfig(ABC):
    """Abstract base class for local model provider configurations.

    This class defines the interface for provider-specific configurations,
    including parameter mapping, environment validation, and URL handling.

    Example:
        >>> class MyProviderConfig(BaseProviderConfig):
        ...     @property
        ...     def provider_type(self) -> ProviderType:
        ...         return ProviderType.OLLAMA
        ...     # Implement other abstract methods...
    """

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type.

        Returns:
            The ProviderType enum value for this provider.
        """
        pass

    @abstractmethod
    def get_supported_openai_params(self, model: str) -> List[str]:
        """Get list of supported OpenAI standard parameters.

        Args:
            model: The model name.

        Returns:
            List of supported parameter names.
        """
        pass

    @abstractmethod
    def map_openai_params(
        self,
        openai_params: Dict[str, Any],
        model: str,
        drop_unsupported: bool = True,
    ) -> Dict[str, Any]:
        """Map OpenAI parameters to provider-specific format.

        Args:
            openai_params: Dictionary of OpenAI format parameters.
            model: The model name being used.
            drop_unsupported: If True, drop unsupported parameters.

        Returns:
            Dictionary of provider-compatible parameters.
        """
        pass

    @abstractmethod
    def get_structured_output_config(
        self,
        structured_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Get configuration for structured output generation.

        Args:
            structured_model: Optional Pydantic model defining the output schema.

        Returns:
            Dictionary containing structured output configuration.
        """
        pass

    @abstractmethod
    def validate_environment(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Validate environment for API calls.

        Args:
            api_key: Optional API key.
            api_base: Optional base URL.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing validated configuration.
        """
        pass

    @abstractmethod
    def get_default_api_base(self) -> Optional[str]:
        """Get the default API base URL.

        Returns:
            Default API base URL or None.
        """
        pass

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> str:
        """Get the complete URL for API calls.

        Args:
            api_base: Optional base URL.
            endpoint: Optional endpoint path.

        Returns:
            Complete URL for the API call.
        """
        pass

    @abstractmethod
    def get_headers(
        self,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Get HTTP headers for API calls.

        Args:
            api_key: Optional API key for authentication.
            extra_headers: Optional additional headers.

        Returns:
            Dictionary of HTTP headers.
        """
        pass

