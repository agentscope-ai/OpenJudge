from typing import Dict, Type

from rm_gallery.core.rm.module import BaseReward


class RewardRegistry:
    """This class is used to register some modules to registry by a repo
    name."""

    # Dictionary mapping reward module names to their corresponding classes
    _registry: Dict[str, Type[BaseReward]] = {}

    @classmethod
    def register(cls, reward_name: str):
        """
        Register a reward module class.

        Args:
            reward_name (str): The name of the reward module, used as an identifier.
            reward_module (Type[BaseReward]): The class of the reward module to be registered.
        """

        def _register(reward_module):
            cls._registry[reward_name] = reward_module
            return reward_module

        return _register

    @classmethod
    def get(cls, reward_name: str) -> Type[BaseReward] | None:
        """
        Retrieve a reward module class by its name.

        Args:
            reward_name (str): The name of the reward module to retrieve.

        Returns:
            Optional[Type[BaseReward]]: The corresponding reward module class if found; otherwise, None.
        """
        return cls._registry.get(reward_name)
