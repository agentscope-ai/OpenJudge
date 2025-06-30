from functools import partial
from typing import Dict, List, Type

from rm_gallery.core.reward.base import BaseReward


class RewardRegistry:
    """A registry management system for reward modules that maps module names to their corresponding implementation classes.

    This class provides a centralized repository for registering and retrieving reward modules by string identifiers.
    Modules can be registered using decorators and later accessed by their string identifiers.

    Attributes:
        _registry: Internal dictionary storing the mapping between reward module names and their classes.
    """

    # Dictionary mapping reward module names to their corresponding classes
    _registry: Dict[str, Type[BaseReward]] = {}

    @classmethod
    def register(cls, reward_name: str):
        """Create a decorator to register a reward module class with a specified identifier.

        The decorator pattern allows classes to be registered while maintaining their original identity.

        Args:
            reward_name: Unique string identifier for the reward module
            reward_module: The BaseReward subclass to be registered

        Returns:
            A decorator function that registers the module when applied to a class
        """

        def _register(reward_module):
            """Internal registration function that stores the module in the registry.

            Args:
                reward_module: The BaseReward subclass to be registered

            Returns:
                The original reward_module class (unchanged)
            """
            cls._registry[reward_name] = reward_module
            return reward_module

        return _register

    @classmethod
    def get(cls, reward_name: str) -> Type[BaseReward] | None:
        """Retrieve a registered reward module class by its identifier.

        Provides safe access to registered modules without raising errors for missing entries.

        Args:
            reward_name: String identifier of the reward module to retrieve

        Returns:
            The corresponding BaseReward subclass if found, None otherwise
        """
        assert reward_name in cls._registry, f"Reward module '{reward_name}' not found"
        module = cls._registry.get(reward_name, None)
        assert module is not None
        return partial(module, name=reward_name)

    @classmethod
    def list(cls) -> List[str]:
        """
        Returns:
            A list of all registered reward modules
        """
        return list(cls._registry.keys())
