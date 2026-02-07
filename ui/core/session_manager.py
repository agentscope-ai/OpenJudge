# -*- coding: utf-8 -*-
"""Session state management for OpenJudge Studio.

Provides utilities for managing Streamlit session state with
namespacing and persistence support.
"""

from typing import Any, Optional

import streamlit as st


class SessionManager:
    """Session state management utilities.

    Provides namespaced access to session state and utilities
    for managing feature-specific state.
    """

    # Global namespace prefix
    GLOBAL_PREFIX = "_oj_"

    @staticmethod
    def get(key: str, default: Any = None, namespace: Optional[str] = None) -> Any:
        """Get a value from session state.

        Args:
            key: The key to retrieve
            default: Default value if not found
            namespace: Optional namespace prefix

        Returns:
            The stored value or default
        """
        full_key = SessionManager._build_key(key, namespace)
        return st.session_state.get(full_key, default)

    @staticmethod
    def set(key: str, value: Any, namespace: Optional[str] = None) -> None:
        """Set a value in session state.

        Args:
            key: The key to set
            value: The value to store
            namespace: Optional namespace prefix
        """
        full_key = SessionManager._build_key(key, namespace)
        st.session_state[full_key] = value

    @staticmethod
    def delete(key: str, namespace: Optional[str] = None) -> bool:
        """Delete a value from session state.

        Args:
            key: The key to delete
            namespace: Optional namespace prefix

        Returns:
            True if the key was deleted, False if it didn't exist
        """
        full_key = SessionManager._build_key(key, namespace)
        if full_key in st.session_state:
            del st.session_state[full_key]
            return True
        return False

    @staticmethod
    def exists(key: str, namespace: Optional[str] = None) -> bool:
        """Check if a key exists in session state.

        Args:
            key: The key to check
            namespace: Optional namespace prefix

        Returns:
            True if the key exists
        """
        full_key = SessionManager._build_key(key, namespace)
        return full_key in st.session_state

    @staticmethod
    def clear_namespace(namespace: str) -> int:
        """Clear all keys in a namespace.

        Args:
            namespace: The namespace to clear

        Returns:
            Number of keys deleted
        """
        prefix = f"{SessionManager.GLOBAL_PREFIX}{namespace}_"
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith(prefix)]

        for key in keys_to_delete:
            del st.session_state[key]

        return len(keys_to_delete)

    @staticmethod
    def get_namespace_keys(namespace: str) -> list[str]:
        """Get all keys in a namespace.

        Args:
            namespace: The namespace to query

        Returns:
            List of keys (without the namespace prefix)
        """
        prefix = f"{SessionManager.GLOBAL_PREFIX}{namespace}_"
        return [k[len(prefix) :] for k in st.session_state.keys() if k.startswith(prefix)]

    @staticmethod
    def _build_key(key: str, namespace: Optional[str] = None) -> str:
        """Build a full session state key.

        Args:
            key: The base key
            namespace: Optional namespace

        Returns:
            The full namespaced key
        """
        if namespace:
            return f"{SessionManager.GLOBAL_PREFIX}{namespace}_{key}"
        return f"{SessionManager.GLOBAL_PREFIX}{key}"

    @staticmethod
    def get_or_init(key: str, factory: callable, namespace: Optional[str] = None) -> Any:
        """Get a value, initializing it if not present.

        Args:
            key: The key to retrieve
            factory: Callable that returns the default value
            namespace: Optional namespace prefix

        Returns:
            The stored or newly initialized value
        """
        full_key = SessionManager._build_key(key, namespace)
        if full_key not in st.session_state:
            st.session_state[full_key] = factory()
        return st.session_state[full_key]

    @staticmethod
    def update(key: str, updater: callable, namespace: Optional[str] = None) -> Any:
        """Update a value using an updater function.

        Args:
            key: The key to update
            updater: Function that takes the current value and returns the new value
            namespace: Optional namespace prefix

        Returns:
            The updated value
        """
        full_key = SessionManager._build_key(key, namespace)
        current = st.session_state.get(full_key)
        new_value = updater(current)
        st.session_state[full_key] = new_value
        return new_value


class FeatureState:
    """State manager scoped to a specific feature.

    Provides convenient access to feature-namespaced session state.

    Example:
        >>> state = FeatureState("grader")
        >>> state.set("threshold", 0.5)
        >>> threshold = state.get("threshold", 0.5)
    """

    def __init__(self, feature_id: str):
        """Initialize feature state manager.

        Args:
            feature_id: The feature ID to use as namespace
        """
        self.feature_id = feature_id

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from this feature's state."""
        return SessionManager.get(key, default, namespace=self.feature_id)

    def set(self, key: str, value: Any) -> None:
        """Set a value in this feature's state."""
        SessionManager.set(key, value, namespace=self.feature_id)

    def delete(self, key: str) -> bool:
        """Delete a value from this feature's state."""
        return SessionManager.delete(key, namespace=self.feature_id)

    def exists(self, key: str) -> bool:
        """Check if a key exists in this feature's state."""
        return SessionManager.exists(key, namespace=self.feature_id)

    def clear(self) -> int:
        """Clear all state for this feature."""
        return SessionManager.clear_namespace(self.feature_id)

    def keys(self) -> list[str]:
        """Get all keys for this feature."""
        return SessionManager.get_namespace_keys(self.feature_id)

    def get_or_init(self, key: str, factory: callable) -> Any:
        """Get a value, initializing it if not present."""
        return SessionManager.get_or_init(key, factory, namespace=self.feature_id)
