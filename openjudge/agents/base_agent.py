# -*- coding: utf-8 -*-
"""Base agent interface for OpenJudge agent evaluation framework."""

import abc
from typing import Any, Dict, Optional

from openjudge.environments.base_env import BaseEnvironment


class BaseAgent(abc.ABC):
    """Base interface for all agents in the OpenJudge evaluation framework.

    This abstract base class defines the contract that all agents must implement
    to be compatible with the evaluation framework.
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize the agent.

        Args:
            name: Unique name for the agent
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs

    @abc.abstractmethod
    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent in the given environment.

        This method is called before the agent runs any tasks. It should handle
        installation of dependencies, configuration, and any other preparation
        needed for the agent to run successfully.

        Args:
            environment: The execution environment for the agent
        """
        pass

    @abc.abstractmethod
    async def run(self, instruction: str, environment: BaseEnvironment) -> Dict[str, Any]:
        """Run the agent with the given instruction.

        Args:
            instruction: The task instruction for the agent to execute
            environment: The execution environment

        Returns:
            Dictionary containing the result of the agent execution.
            Expected keys:
            - 'success': bool indicating if the task was successful
            - 'output': str containing the agent's output
            - 'metrics': dict containing performance metrics
        """
        pass

    @abc.abstractmethod
    async def teardown(self, environment: BaseEnvironment) -> None:
        """Cleanup after the agent has finished running.

        Args:
            environment: The execution environment
        """
        pass

    @property
    @abc.abstractmethod
    def version(self) -> Optional[str]:
        """Return the version of the agent, if available."""
        pass


class BaseInstalledAgent(BaseAgent):
    """Base class for agents that need to be installed in the environment.

    This provides common functionality for agents that require installation
    of dependencies or packages in the execution environment.
    """

    async def setup(self, environment: BaseEnvironment) -> None:
        """Default setup that installs the agent in the environment."""
        await self.install(environment)

    async def install(self, environment: BaseEnvironment) -> None:
        """Install the agent in the environment.

        This method should be implemented by subclasses to install the agent
        and its dependencies in the execution environment.

        Args:
            environment: The execution environment
        """
        raise NotImplementedError("Subclasses must implement install method")

    async def run(self, instruction: str, environment: BaseEnvironment) -> Dict[str, Any]:
        """Default run implementation that executes the agent command."""
        raise NotImplementedError("Subclasses must implement run method")

    async def teardown(self, environment: BaseEnvironment) -> None:
        """Default teardown that does nothing."""
        pass