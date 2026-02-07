# -*- coding: utf-8 -*-
"""Navigation management for OpenJudge Studio.

Handles the navigation UI and routing between feature modules.
"""

from typing import Optional

import streamlit as st
from core.base_feature import BaseFeature
from core.feature_registry import FeatureRegistry
from shared.i18n import t

# Session state key for current feature
CURRENT_FEATURE_KEY = "_current_feature_id"
PREVIOUS_FEATURE_KEY = "_previous_feature_id"


class Navigation:
    """Navigation manager for feature routing.

    Handles rendering the navigation UI and managing the current
    feature selection state.
    """

    @staticmethod
    def render_feature_selector() -> str:
        """Render the feature selection UI in the sidebar.

        Displays a selectbox (dropdown) for selecting between features.

        Returns:
            The ID of the selected feature
        """
        features = FeatureRegistry.get_all()

        if not features:
            st.warning("No features registered")
            return ""

        # Build options - use stable feature_ids (not translated labels)
        feature_ids = [f.feature_id for f in features]

        # Get default feature id
        default_id = FeatureRegistry.get_default_feature_id()

        # Widget key for selectbox
        widget_key = "_nav_feature_selector"

        # Initialize widget state if not exists
        if widget_key not in st.session_state:
            st.session_state[widget_key] = default_id

        # Ensure current value is valid (in case features changed)
        if st.session_state[widget_key] not in feature_ids:
            st.session_state[widget_key] = default_id

        # Get previous value for lifecycle hooks
        previous_id = st.session_state.get(CURRENT_FEATURE_KEY)

        # Build labels dynamically (these change with language, but values stay stable)
        feature_labels = {f.feature_id: FeatureRegistry.get_instance(f.feature_id).display_label for f in features}

        # Render selectbox with stable feature_ids as options
        # No index parameter - let Streamlit manage state via key
        selected_id = st.selectbox(
            t("app.features"),
            options=feature_ids,
            format_func=lambda x: feature_labels.get(x, x),
            key=widget_key,
            label_visibility="collapsed",
        )

        # Track feature changes for lifecycle hooks
        if previous_id is not None and selected_id != previous_id:
            st.session_state[PREVIOUS_FEATURE_KEY] = previous_id

        # Always update current feature id
        st.session_state[CURRENT_FEATURE_KEY] = selected_id

        return selected_id

    @staticmethod
    def get_current_feature_id() -> Optional[str]:
        """Get the currently selected feature ID.

        Returns:
            The current feature ID, or None if not set
        """
        return st.session_state.get(CURRENT_FEATURE_KEY)

    @staticmethod
    def get_previous_feature_id() -> Optional[str]:
        """Get the previously selected feature ID.

        Useful for lifecycle hooks when switching features.

        Returns:
            The previous feature ID, or None
        """
        return st.session_state.get(PREVIOUS_FEATURE_KEY)

    @staticmethod
    def get_current_feature() -> Optional[BaseFeature]:
        """Get an instance of the currently selected feature.

        Returns:
            A new instance of the current feature, or None
        """
        feature_id = Navigation.get_current_feature_id()
        if feature_id:
            return FeatureRegistry.get_instance(feature_id)
        return None

    @staticmethod
    def set_current_feature(feature_id: str) -> bool:
        """Programmatically set the current feature.

        Args:
            feature_id: The ID of the feature to switch to

        Returns:
            True if successful, False if feature not found
        """
        if FeatureRegistry.is_registered(feature_id):
            current = st.session_state.get(CURRENT_FEATURE_KEY)
            if current != feature_id:
                st.session_state[PREVIOUS_FEATURE_KEY] = current
                st.session_state[CURRENT_FEATURE_KEY] = feature_id
            return True
        return False

    @staticmethod
    def has_feature_changed() -> bool:
        """Check if the feature was just changed.

        Returns:
            True if the feature was changed in this interaction
        """
        current = st.session_state.get(CURRENT_FEATURE_KEY)
        previous = st.session_state.get(PREVIOUS_FEATURE_KEY)
        return previous is not None and current != previous

    @staticmethod
    def clear_feature_change() -> None:
        """Clear the feature change tracking.

        Call this after handling lifecycle hooks.
        """
        if PREVIOUS_FEATURE_KEY in st.session_state:
            del st.session_state[PREVIOUS_FEATURE_KEY]
