# -*- coding: utf-8 -*-
"""Base feature class for OpenJudge Studio.

All feature modules must inherit from BaseFeature and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any

import streamlit as st


class BaseFeature(ABC):
    """Abstract base class for all feature modules.

    Each feature module (Grader, Auto Arena, AutoRubric, etc.) must inherit
    from this class and implement the required methods.

    Class Attributes:
        feature_id: Unique identifier for the feature (e.g., "grader", "auto_arena")
        feature_name: Display name for the feature (e.g., "Grader 评估")
        feature_icon: Icon emoji for the feature (e.g., "⚖️")
        feature_description: Brief description of the feature
        order: Display order in navigation (lower = higher priority)

    Example:
        >>> class GraderFeature(BaseFeature):
        ...     feature_id = "grader"
        ...     feature_name = "Grader 评估"
        ...     feature_icon = "⚖️"
        ...     feature_description = "使用内置 Grader 对单条数据进行评估"
        ...     order = 1
        ...
        ...     def render_sidebar(self) -> dict:
        ...         # Render sidebar configuration
        ...         return {"key": "value"}
        ...
        ...     def render_main_content(self, sidebar_config: dict) -> None:
        ...         # Render main content area
        ...         st.write("Main content")
    """

    # Required class attributes (must be defined in subclasses)
    feature_id: str
    feature_name: str
    feature_icon: str
    feature_description: str
    order: int = 100  # Default order (lower = higher priority)

    def __init__(self) -> None:
        """Initialize the feature instance."""
        # Validate required class attributes
        required_attrs = ["feature_id", "feature_name", "feature_icon", "feature_description"]
        for attr in required_attrs:
            if not hasattr(self, attr) or not getattr(self, attr):
                raise NotImplementedError(
                    f"Feature class {self.__class__.__name__} must define '{attr}' class attribute"
                )

    @abstractmethod
    def render_sidebar(self) -> dict[str, Any]:
        """Render the sidebar configuration panel.

        This method should render all sidebar UI elements specific to this feature
        and return a dictionary containing the configuration values.

        Returns:
            Dictionary containing all sidebar configuration values.
            This will be passed to render_main_content().
        """

    @abstractmethod
    def render_main_content(self, sidebar_config: dict[str, Any]) -> None:
        """Render the main content area.

        This method should render the main UI for this feature based on
        the configuration from the sidebar.

        Args:
            sidebar_config: Configuration dictionary from render_sidebar()
        """

    def render_header(self) -> None:
        """Render the feature header.

        Override this method to customize the header for your feature.
        Default implementation renders a simple title with icon.
        """
        st.markdown(
            f"""<div style="margin-bottom: 1rem;">
                <h1 style="
                    font-size: 2rem;
                    font-weight: 700;
                    color: #F1F5F9;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                ">
                    <span>{self.feature_icon}</span>
                    <span>{self.feature_name}</span>
                </h1>
                <p style="color: #94A3B8; margin: 0.25rem 0 0 0; font-size: 0.95rem;">
                    {self.feature_description}
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    def on_mount(self) -> None:
        """Called when the feature is mounted (selected).

        Override this method to perform initialization when the feature
        becomes active. Useful for loading saved state, etc.
        """

    def on_unmount(self) -> None:
        """Called when the feature is unmounted (deselected).

        Override this method to perform cleanup when switching away
        from this feature. Useful for saving state, etc.
        """

    def get_session_key(self, key: str) -> str:
        """Get a namespaced session state key for this feature.

        This ensures session state keys don't conflict between features.

        Args:
            key: The base key name

        Returns:
            Namespaced key: f"{feature_id}_{key}"
        """
        return f"{self.feature_id}_{key}"

    def get_session_value(self, key: str, default: Any = None) -> Any:
        """Get a value from session state with feature namespace.

        Args:
            key: The base key name
            default: Default value if key doesn't exist

        Returns:
            The value from session state, or default
        """
        full_key = self.get_session_key(key)
        return st.session_state.get(full_key, default)

    def set_session_value(self, key: str, value: Any) -> None:
        """Set a value in session state with feature namespace.

        Args:
            key: The base key name
            value: The value to store
        """
        full_key = self.get_session_key(key)
        st.session_state[full_key] = value

    @property
    def display_label(self) -> str:
        """Get the display label for navigation.

        Returns:
            Formatted string with icon and name
        """
        return f"{self.feature_icon} {self.feature_name}"
