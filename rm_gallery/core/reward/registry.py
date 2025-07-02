from typing import Dict, Type

import pandas as pd
from tabulate import tabulate

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
    def register(cls, name: str):
        """Create a decorator to register a reward module class with a specified identifier.

        The decorator pattern allows classes to be registered while maintaining their original identity.

        Args:
            name: Unique string identifier for the reward module
            module: The BaseReward subclass to be registered

        Returns:
            A decorator function that registers the module when applied to a class
        """

        def _register(module: Type[BaseReward]):
            """Internal registration function that stores the module in the registry.

            Args:
                module: The BaseReward subclass to be registered

            Returns:
                The original module class (unchanged)
            """
            cls._registry[name] = module
            return module

        return _register

    @classmethod
    def get(cls, name: str) -> Type[BaseReward] | None:
        """Retrieve a registered reward module class by its identifier.

        Provides safe access to registered modules without raising errors for missing entries.

        Args:
            name: String identifier of the reward module to retrieve

        Returns:
            The corresponding BaseReward subclass if found, None otherwise
        """
        assert name in cls._registry, f"Reward module '{name}' not found"
        return cls._registry.get(name, None)

    @classmethod
    def list(cls) -> str:
        """
        Returns:
            A list of all registered reward modules
        """
        info = []
        for name, module in cls._registry.items():
            info.append(
                pd.Series(
                    {
                        "Name": name,
                        "Class": module.__name__,
                        "Scenario": module.__doc__.strip(),
                    }
                )
            )

        info_df = pd.concat(info, axis=1).T
        # info_str = info_df.to_markdown(index=False)
        info_str = tabulate(
            info_df,
            headers="keys",
            tablefmt="grid",
            maxcolwidths=[50] * (len(info_df.columns) + 1),
            # showindex=False,
        )
        # info_str = tabulate(info_df, headers='keys', tablefmt='github')
        return info_str
