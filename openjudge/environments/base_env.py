# -*- coding: utf-8 -*-
"""Base environment interface for OpenJudge agent evaluation framework."""

import abc
from pathlib import Path
from typing import Any, Dict, Optional


class BaseEnvironment(abc.ABC):
    """Base interface for execution environments in the OpenJudge framework.

    This abstract base class defines the contract that all execution environments
    must implement to be compatible with the evaluation framework.
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize the environment.

        Args:
            name: Unique name for the environment
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs

    @abc.abstractmethod
    async def start(self) -> None:
        """Start the execution environment."""
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the execution environment."""
        pass

    @abc.abstractmethod
    async def execute_command(self, command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute a command in the environment.

        Args:
            command: The command to execute
            timeout: Optional timeout in seconds

        Returns:
            Dictionary containing:
            - 'stdout': Standard output
            - 'stderr': Standard error
            - 'returncode': Return code
            - 'success': Boolean indicating success
        """
        pass

    @abc.abstractmethod
    async def copy_to(self, source: Path, destination: str) -> bool:
        """Copy a file from host to environment.

        Args:
            source: Source path on the host
            destination: Destination path in the environment

        Returns:
            Boolean indicating success
        """
        pass

    @abc.abstractmethod
    async def copy_from(self, source: str, destination: Path) -> bool:
        """Copy a file from environment to host.

        Args:
            source: Source path in the environment
            destination: Destination path on the host

        Returns:
            Boolean indicating success
        """
        pass

    @abc.abstractmethod
    async def write_file(self, path: str, content: str) -> bool:
        """Write content to a file in the environment.

        Args:
            path: Path in the environment
            content: Content to write

        Returns:
            Boolean indicating success
        """
        pass

    @abc.abstractmethod
    async def read_file(self, path: str) -> Optional[str]:
        """Read content from a file in the environment.

        Args:
            path: Path in the environment

        Returns:
            File content as string, or None if file doesn't exist
        """
        pass

    @property
    @abc.abstractmethod
    def is_running(self) -> bool:
        """Check if the environment is currently running."""
        pass