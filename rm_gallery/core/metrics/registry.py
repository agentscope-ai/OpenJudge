"""
Grader Registry

Registry for managing and discovering all available graders.
Uses singleton pattern to ensure global uniqueness.
"""

from typing import Callable, Dict, List, Optional

from loguru import logger

from rm_gallery.core.grader import Grader


class GraderRegistry:
    """
    Grader Registry (Singleton Pattern)

    Manages all registered graders, provides registration, query and listing functionality.

    Example:
        >>> from rm_gallery.core.metrics import grader_registry, register_grader
        >>>
        >>> @register_grader("my_grader")
        ... class MyGrader(Grader):
        ...     async def evaluate(self, **kwargs):
        ...         return GraderScore(score=1.0, reason="Test")
        >>>
        >>> # Get registered grader
        >>> GraderClass = grader_registry.get("my_grader")
        >>> grader = GraderClass()
    """

    _instance: Optional["GraderRegistry"] = None
    _graders: Dict[str, type[Grader] | Callable] = {}

    def __new__(cls) -> "GraderRegistry":
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._graders = {}
        return cls._instance

    def register(
        self, name: str, grader_class: type[Grader] | Callable, override: bool = False
    ) -> None:
        """
        Register grader class

        Args:
            name: Grader name (unique identifier)
            grader_class: Grader class (must inherit from Grader or be a callable)
            override: Whether to allow overriding existing graders

        Raises:
            ValueError: When grader already exists and override=False

        Example:
            >>> registry = GraderRegistry()
            >>> registry.register("bleu", BLEUGrader)
        """
        # Check if already exists
        if name in self._graders and not override:
            logger.warning(
                f"Grader '{name}' is already registered. "
                f"Use override=True to replace it, or choose a different name."
            )
            return

        self._graders[name] = grader_class
        logger.info(f"Registered grader: {name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister grader

        Args:
            name: Grader name

        Returns:
            bool: Whether successfully unregistered

        Example:
            >>> registry.unregister("bleu")
            True
        """
        if name in self._graders:
            del self._graders[name]
            logger.info(f"Unregistered grader: {name}")
            return True
        else:
            logger.warning(f"Grader '{name}' not found in registry")
            return False

    def get(self, name: str) -> Optional[type[Grader] | Callable]:
        """
        Get grader class

        Args:
            name: Grader name

        Returns:
            Optional[type[Grader] | Callable]: Grader class, returns None if not found

        Example:
            >>> GraderClass = registry.get("bleu")
            >>> if GraderClass:
            ...     grader = GraderClass()
        """
        grader_class = self._graders.get(name)
        if grader_class is None:
            logger.warning(f"Grader '{name}' not found in registry")
        return grader_class

    def get_instance(self, name: str, **kwargs) -> Optional[Grader]:
        """
        Get grader instance

        Args:
            name: Grader name
            **kwargs: Parameters passed to grader constructor

        Returns:
            Optional[Grader]: Grader instance

        Example:
            >>> grader = registry.get_instance("bleu", max_ngram_order=4)
        """
        grader_class = self.get(name)
        if grader_class is None:
            return None

        try:
            return grader_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating instance of grader '{name}': {e}")
            return None

    def list_graders(self) -> List[str]:
        """
        List all registered grader names

        Returns:
            List[str]: List of grader names

        Example:
            >>> registry.list_graders()
            ['bleu', 'rouge', 'exact_match', 'substring_match']
        """
        return sorted(list(self._graders.keys()))

    def list_graders_by_category(self) -> Dict[str, List[str]]:
        """
        List graders by category

        Returns:
            Dict[str, List[str]]: Mapping from category to list of grader names

        Example:
            >>> registry.list_graders_by_category()
            {
                'nlp_metrics': ['bleu', 'rouge', 'meteor'],
                'text_similarity': ['fuzzy_match', 'cosine'],
                'string_check': ['exact_match', 'substring']
            }
        """
        categories: Dict[str, List[str]] = {}

        for name, grader_class in self._graders.items():
            # Infer category from module path
            if hasattr(grader_class, "__module__"):
                module_path = grader_class.__module__
                if "nlp_metrics" in module_path:
                    category = "nlp_metrics"
                elif "text_similarity" in module_path:
                    category = "text_similarity"
                elif "string_check" in module_path:
                    category = "string_check"
                elif "format_check" in module_path:
                    category = "format_check"
                else:
                    category = "other"
            else:
                category = "other"

            if category not in categories:
                categories[category] = []
            categories[category].append(name)

        return categories

    def is_registered(self, name: str) -> bool:
        """
        Check if grader is registered

        Args:
            name: Grader name

        Returns:
            bool: Whether registered
        """
        return name in self._graders

    def clear(self) -> None:
        """
        Clear registry (mainly for testing)

        Warning:
            This operation removes all registered graders, use with caution.
        """
        count = len(self._graders)
        self._graders.clear()
        logger.warning(f"Cleared grader registry ({count} graders removed)")

    def __len__(self) -> int:
        """Return number of registered graders"""
        return len(self._graders)

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator"""
        return name in self._graders

    def __repr__(self) -> str:
        """String representation"""
        return f"GraderRegistry({len(self._graders)} graders registered)"


# Global registry instance
grader_registry = GraderRegistry()


def register_grader(name: str, override: bool = False):
    """
    Decorator: Register grader class

    This is the most concise way to register graders.

    Args:
        name: Grader name
        override: Whether to allow overriding existing graders

    Example:
        >>> @register_grader("exact_match")
        ... class ExactMatchGrader(Grader):
        ...     async def evaluate(self, reference: str, candidate: str, **kwargs):
        ...         matched = reference == candidate
        ...         return GraderScore(score=1.0 if matched else 0.0, reason="Exact match")
    """

    def decorator(cls: type[Grader] | Callable) -> type[Grader] | Callable:
        grader_registry.register(name, cls, override=override)
        return cls

    return decorator


def get_grader(name: str, **kwargs) -> Optional[Grader]:
    """
    Helper function: Get grader instance

    Args:
        name: Grader name
        **kwargs: Parameters passed to grader constructor

    Returns:
        Optional[Grader]: Grader instance

    Example:
        >>> bleu = get_grader("bleu", max_ngram_order=4)
        >>> if bleu:
        ...     result = await bleu.evaluate(reference="test", candidate="test")
    """
    return grader_registry.get_instance(name, **kwargs)


def list_available_graders() -> List[str]:
    """
    Helper function: List all available graders

    Returns:
        List[str]: List of grader names

    Example:
        >>> graders = list_available_graders()
        >>> print(f"Available graders: {', '.join(graders)}")
    """
    return grader_registry.list_graders()
