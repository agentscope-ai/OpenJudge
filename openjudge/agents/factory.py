# -*- coding: utf-8 -*-
"""Factory for creating agent instances."""

from typing import TYPE_CHECKING, Dict, Type

from openjudge.agents.base_agent import BaseAgent
from openjudge.agents.name import AgentName

if TYPE_CHECKING:
    from openjudge.agents.installed.openclaw import OpenClaw
    from openjudge.agents.installed.copaw import Copaw


class AgentFactory:
    """Factory for creating agent instances."""

    _AGENTS: Dict[AgentName, Type[BaseAgent]] = {}

    @classmethod
    def register_agent(cls, name: AgentName, agent_class: Type[BaseAgent]) -> None:
        """Register an agent class with the factory."""
        cls._AGENTS[name] = agent_class

    @classmethod
    def create_agent(cls, name: AgentName, **kwargs) -> BaseAgent:
        """Create an agent instance.

        Args:
            name: The name of the agent to create
            **kwargs: Additional arguments to pass to the agent constructor

        Returns:
            An instance of the requested agent

        Raises:
            ValueError: If the agent name is not registered
        """
        if name not in cls._AGENTS:
            raise ValueError(f"Unknown agent type: {name}. Available agents: {list(cls._AGENTS.keys())}")

        agent_class = cls._AGENTS[name]
        return agent_class(name.value, **kwargs)

    @classmethod
    def get_available_agents(cls) -> Dict[AgentName, Type[BaseAgent]]:
        """Get all registered agents.

        Returns:
            Dictionary mapping agent names to their classes
        """
        return cls._AGENTS.copy()


# Import and register agents after the factory is defined to avoid circular imports
def _register_agents():
    """Register all built-in agents."""
    try:
        from openjudge.agents.installed.openclaw import OpenClaw
        AgentFactory.register_agent(AgentName.OPENCLAW, OpenClaw)
    except ImportError:
        pass

    try:
        from openjudge.agents.installed.copaw import Copaw
        AgentFactory.register_agent(AgentName.COPAW, Copaw)
    except ImportError:
        pass

    # Add more agents as they are implemented


# Register agents when the module is imported
_register_agents()