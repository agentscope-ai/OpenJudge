# -*- coding: utf-8 -*-
"""Feature registry for OpenJudge Studio.

Provides a central registry for managing feature modules.
Features are registered at application startup and can be
dynamically added or removed.
"""

from typing import Optional

from core.base_feature import BaseFeature


class FeatureRegistry:
    """Central registry for feature modules.

    This class manages the registration and retrieval of feature modules.
    It's implemented as a class with class methods for simplicity, acting
    as a singleton registry.

    Features are cached as singleton instances to maintain state across
    Streamlit reruns and enable proper lifecycle management.

    Example:
        >>> from features.grader import GraderFeature
        >>> FeatureRegistry.register(GraderFeature)
        >>> feature = FeatureRegistry.get_instance("grader")
    """

    _features: dict[str, type[BaseFeature]] = {}
    _instances: dict[str, BaseFeature] = {}  # Singleton instance cache

    @classmethod
    def register(cls, feature_class: type[BaseFeature], replace: bool = False) -> None:
        """Register a feature module.

        This method is idempotent - registering the same class twice is safe.
        This is important for Streamlit's script rerun model.

        Args:
            feature_class: The feature class to register (must be a subclass of BaseFeature)
            replace: If True, replace existing registration. If False (default),
                     skip if the same class is already registered, raise if different class.

        Raises:
            TypeError: If feature_class is not a subclass of BaseFeature
            ValueError: If a different feature class with the same ID is already registered
                        and replace=False
        """
        if not isinstance(feature_class, type) or not issubclass(feature_class, BaseFeature):
            raise TypeError(f"feature_class must be a subclass of BaseFeature, got {type(feature_class)}")

        feature_id = feature_class.feature_id

        if feature_id in cls._features:
            existing_class = cls._features[feature_id]
            # Same class - idempotent, just return
            if existing_class is feature_class:
                return
            # Different class with same ID
            if not replace:
                raise ValueError(
                    f"Feature with ID '{feature_id}' is already registered "
                    f"with a different class ({existing_class.__name__})"
                )
            # Clear cached instance when replacing with different class
            cls._instances.pop(feature_id, None)

        cls._features[feature_id] = feature_class

    @classmethod
    def unregister(cls, feature_id: str) -> bool:
        """Unregister a feature module and remove its cached instance.

        Args:
            feature_id: The ID of the feature to unregister

        Returns:
            True if the feature was unregistered, False if it wasn't registered
        """
        if feature_id in cls._features:
            del cls._features[feature_id]
            # Also remove cached instance
            cls._instances.pop(feature_id, None)
            return True
        return False

    @classmethod
    def get(cls, feature_id: str) -> Optional[type[BaseFeature]]:
        """Get a feature class by ID.

        Args:
            feature_id: The ID of the feature to retrieve

        Returns:
            The feature class, or None if not found
        """
        return cls._features.get(feature_id)

    @classmethod
    def get_instance(cls, feature_id: str) -> Optional[BaseFeature]:
        """Get a cached feature instance by ID.

        Instances are cached as singletons to maintain state across
        Streamlit reruns and enable proper lifecycle management.

        Args:
            feature_id: The ID of the feature

        Returns:
            The cached feature instance, or None if not found
        """
        # Return cached instance if exists
        if feature_id in cls._instances:
            return cls._instances[feature_id]

        # Create and cache new instance
        feature_class = cls.get(feature_id)
        if feature_class:
            instance = feature_class()
            cls._instances[feature_id] = instance
            return instance
        return None

    @classmethod
    def get_all(cls) -> list[type[BaseFeature]]:
        """Get all registered feature classes.

        Returns:
            List of feature classes sorted by their order attribute
        """
        return sorted(cls._features.values(), key=lambda f: f.order)

    @classmethod
    def get_all_ids(cls) -> list[str]:
        """Get all registered feature IDs.

        Returns:
            List of feature IDs sorted by their order attribute
        """
        return [f.feature_id for f in cls.get_all()]

    @classmethod
    def get_default_feature_id(cls) -> Optional[str]:
        """Get the default (first) feature ID.

        Returns:
            The ID of the first feature by order, or None if no features registered
        """
        features = cls.get_all()
        if features:
            return features[0].feature_id
        return None

    @classmethod
    def is_registered(cls, feature_id: str) -> bool:
        """Check if a feature is registered.

        Args:
            feature_id: The ID to check

        Returns:
            True if the feature is registered
        """
        return feature_id in cls._features

    @classmethod
    def clear(cls) -> None:
        """Clear all registered features and cached instances.

        Mainly useful for testing.
        """
        cls._features.clear()
        cls._instances.clear()

    @classmethod
    def count(cls) -> int:
        """Get the number of registered features.

        Returns:
            Number of registered features
        """
        return len(cls._features)
