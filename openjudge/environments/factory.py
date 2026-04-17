# -*- coding: utf-8 -*-
"""Factory for creating environment instances."""

from typing import TYPE_CHECKING, Dict, Optional, Type

from openjudge.environments.base_env import BaseEnvironment

if TYPE_CHECKING:
    from openjudge.environments.docker_env import DockerEnvironment


class EnvironmentFactory:
    """Factory for creating environment instances."""

    _ENVIRONMENTS: Dict[str, Type[BaseEnvironment]] = {}

    @classmethod
    def register_environment(cls, name: str, env_class: Type[BaseEnvironment]) -> None:
        """Register an environment class with the factory."""
        cls._ENVIRONMENTS[name] = env_class

    @classmethod
    def create_environment(cls, name: str, **kwargs) -> BaseEnvironment:
        """Create an environment instance.

        Args:
            name: The name/type of the environment to create
            **kwargs: Additional arguments to pass to the environment constructor

        Returns:
            An instance of the requested environment

        Raises:
            ValueError: If the environment type is not registered
        """
        if name not in cls._ENVIRONMENTS:
            raise ValueError(f"Unknown environment type: {name}. Available environments: {list(cls._ENVIRONMENTS.keys())}")

        env_class = cls._ENVIRONMENTS[name]
        return env_class(name, **kwargs)

    @classmethod
    def get_available_environments(cls) -> Dict[str, Type[BaseEnvironment]]:
        """Get all registered environments.

        Returns:
            Dictionary mapping environment names to their classes
        """
        return cls._ENVIRONMENTS.copy()


# Register built-in environments
def _register_environments():
    """Register all built-in environments."""
    try:
        from openjudge.environments.docker_env import DockerEnvironment
        EnvironmentFactory.register_environment("docker", DockerEnvironment)
    except ImportError:
        pass


# Register environments when the module is imported
_register_environments()