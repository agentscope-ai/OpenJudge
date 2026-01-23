# -*- coding: utf-8 -*-
"""
OpenJudge Studio - A modern LLM evaluation interface.

This is the main entry point for the Streamlit application.
The application uses a modular architecture with multiple feature modules
that can be switched via the navigation system.
"""

import sys
from pathlib import Path

# Add the ui directory to Python path for local imports
UI_DIR = Path(__file__).parent
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

# pylint: disable=wrong-import-position
import streamlit as st  # noqa: E402
from core.feature_registry import FeatureRegistry  # noqa: E402
from core.navigation import Navigation  # noqa: E402

# Import feature modules
from features.grader import GraderFeature  # noqa: E402
from features.zero_shot import ZeroShotFeature  # noqa: E402
from shared.components.common import render_footer  # noqa: E402
from shared.components.logo import render_logo_and_title  # noqa: E402
from shared.i18n import render_language_selector, t  # noqa: E402
from shared.styles.theme import inject_css  # noqa: E402

# pylint: enable=wrong-import-position

# ============================================================================
# Feature Registration
# ============================================================================

# Register all available features
# Add new features here as they are implemented
FeatureRegistry.register(GraderFeature)
FeatureRegistry.register(ZeroShotFeature)
# Future features:
# from features.autorubric import AutoRubricFeature
# FeatureRegistry.register(AutoRubricFeature)

# ============================================================================
# Page Configuration (must be first Streamlit command)
# ============================================================================
st.set_page_config(
    page_title="OpenJudge Studio",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main function to run the OpenJudge Studio application."""
    # Inject custom CSS
    inject_css()

    # ========================================================================
    # Sidebar Configuration
    # ========================================================================
    with st.sidebar:
        # Logo and title
        render_logo_and_title()

        # Divider
        st.markdown('<div class="custom-divider" style="margin: 0.75rem 0;"></div>', unsafe_allow_html=True)

        # Language selector
        render_language_selector()

        # Divider
        st.markdown('<div class="custom-divider" style="margin: 0.75rem 0;"></div>', unsafe_allow_html=True)

        # Feature navigation
        st.markdown(
            f'<div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; '
            f'letter-spacing: 0.1em; color: #64748B; margin-bottom: 0.5rem;">{t("app.features")}</div>',
            unsafe_allow_html=True,
        )
        selected_feature_id = Navigation.render_feature_selector()

        # Divider before feature-specific config
        st.markdown('<div class="custom-divider" style="margin: 0.75rem 0;"></div>', unsafe_allow_html=True)

        # Get current feature instance
        feature = FeatureRegistry.get_instance(selected_feature_id)

        # Render feature-specific sidebar config
        sidebar_config = {}
        if feature:
            sidebar_config = feature.render_sidebar()

    # ========================================================================
    # Feature Lifecycle Management
    # ========================================================================

    # Handle feature switching - call on_unmount for previous feature
    if Navigation.has_feature_changed():
        previous_feature_id = Navigation.get_previous_feature_id()
        if previous_feature_id:
            previous_feature = FeatureRegistry.get_instance(previous_feature_id)
            if previous_feature:
                previous_feature.on_unmount()
        # Clear the change tracking after handling
        Navigation.clear_feature_change()

    # ========================================================================
    # Main Content Area
    # ========================================================================

    if feature:
        # Call lifecycle hook for current feature
        feature.on_mount()

        # Render feature header
        feature.render_header()

        # Render feature main content
        feature.render_main_content(sidebar_config)

    else:
        # No feature selected (shouldn't happen normally)
        st.warning(t("app.no_feature_selected"))

    # ========================================================================
    # Footer
    # ========================================================================
    render_footer()


if __name__ == "__main__":
    main()
