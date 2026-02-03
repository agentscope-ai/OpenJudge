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
from features.auto_arena import AutoArenaFeature  # noqa: E402
from features.auto_rubric import AutoRubricFeature  # noqa: E402

# Import feature modules
from features.grader import GraderFeature  # noqa: E402
from features.paper_review import PaperReviewFeature  # noqa: E402
from shared.components.common import render_footer  # noqa: E402
from shared.components.logo import render_logo_and_title  # noqa: E402
from shared.components.workspace_selector import (  # noqa: E402
    ENABLE_SHARED_WORKSPACES,
    render_workspace_selector,
)
from shared.i18n import (  # noqa: E402
    inject_language_loader,
    render_language_selector,
    t,
)
from shared.services.workspace_manager import (  # noqa: E402
    get_storage_manager,
    initialize_workspace_from_url,
    inject_browser_id_loader,
)
from shared.styles.theme import inject_css  # noqa: E402

# pylint: enable=wrong-import-position

# ============================================================================
# Feature Registration
# ============================================================================

# Register all available features
# Add new features here as they are implemented
FeatureRegistry.register(GraderFeature)
FeatureRegistry.register(AutoArenaFeature)
FeatureRegistry.register(PaperReviewFeature)
FeatureRegistry.register(AutoRubricFeature)

# ============================================================================
# Page Configuration (must be first Streamlit command)
# ============================================================================
st.set_page_config(
    page_title="OpenJudge Studio",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _check_storage_cleanup() -> None:
    """Check and perform storage cleanup if needed.

    This runs once per session to avoid performance impact.
    Cleanup is triggered if:
    - Storage exceeds 500MB
    - Data is older than 30 days
    """
    # Only check once per session
    if st.session_state.get("_storage_cleanup_checked"):
        return

    st.session_state["_storage_cleanup_checked"] = True

    try:
        storage_mgr = get_storage_manager()
        cleanup_result = storage_mgr.auto_cleanup_if_needed(
            max_mb=500,  # 500 MB per workspace
            retention_days=30,  # Keep data for 30 days
        )

        if cleanup_result and cleanup_result["deleted_dirs"] > 0:
            # Log cleanup but don't show to user (happens silently)
            from loguru import logger

            logger.info(
                f"Auto-cleanup: deleted {cleanup_result['deleted_dirs']} old items, "
                f"freed {cleanup_result['freed_mb']} MB"
            )
    except Exception:
        # Silently ignore cleanup errors
        pass


def main() -> None:
    """Main function to run the OpenJudge Studio application."""
    # Inject custom CSS
    inject_css()

    # Initialize workspace from URL (doesn't render anything)
    initialize_workspace_from_url()

    # Periodic storage cleanup check (runs once per session)
    _check_storage_cleanup()

    # ========================================================================
    # Sidebar Configuration
    # ========================================================================
    with st.sidebar:
        # Initialize browser ID (inside sidebar to avoid main area spacing)
        inject_browser_id_loader()

        # Load language preference
        inject_language_loader()

        # Logo and title
        render_logo_and_title()

        # Divider
        st.markdown('<div class="custom-divider" style="margin: 0.75rem 0;"></div>', unsafe_allow_html=True)

        # Workspace selector (only if shared workspaces are enabled)
        if ENABLE_SHARED_WORKSPACES:
            # Workspace selector with language selector in the same row
            render_workspace_selector(show_language_selector=True)
        else:
            # Just show language selector when workspace selector is hidden
            render_language_selector(position="sidebar")

        # Divider after workspace/language selector
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
