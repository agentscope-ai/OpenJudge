from typing import Dict, Type, Optional, List

from rm_gallery.core.rm.module import BaseRewardModule


class GalleryRegistry:
    """Registry for reward modules, used to manage evaluators for different scoring systems."""
    
    # Dictionary mapping reward module names to their corresponding classes
    _registry: Dict[str, Type[BaseRewardModule]] = {}
    
    @classmethod
    def register(cls, reward_name: str, reward_module: Type[BaseRewardModule]):
        """
        Register a reward module class.

        Args:
            reward_name (str): The name of the reward module, used as an identifier.
            reward_module (Type[BaseRewardModule]): The class of the reward module to be registered.
        """
        cls._registry[reward_name] = reward_module
    
    @classmethod
    def get_module(cls, reward_name: str) -> Optional[Type[BaseRewardModule]]:
        """
        Retrieve a reward module class by its name.

        Args:
            reward_name (str): The name of the reward module to retrieve.

        Returns:
            Optional[Type[BaseRewardModule]]: The corresponding reward module class if found; otherwise, None.
        """
        return cls._registry.get(reward_name)
