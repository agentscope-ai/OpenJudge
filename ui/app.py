# -*- coding: utf-8 -*-
"""
OpenJudge Studio - A modern LLM evaluation interface.

This is the main entry point for the Streamlit application.
The application uses the OpenJudge framework's built-in Graders and Models.
"""

import sys
from pathlib import Path

# Add the ui directory to Python path for local imports
UI_DIR = Path(__file__).parent
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

# pylint: disable=wrong-import-position
import streamlit as st  # noqa: E402
from components.input_panel import render_input_panel, render_run_button  # noqa: E402
from components.result_panel import render_result_panel  # noqa: E402
from components.shared import (  # noqa: E402
    render_divider,
    render_footer,
    render_header,
    render_quick_guide,
)
from components.sidebar import render_sidebar  # noqa: E402
from config.constants import APP_NAME  # noqa: E402
from styles.theme import inject_css  # noqa: E402

# pylint: enable=wrong-import-position

# ============================================================================
# Page Configuration (must be first Streamlit command)
# ============================================================================
st.set_page_config(
    page_title=APP_NAME,
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
    sidebar_config = render_sidebar()

    # ========================================================================
    # Main Content Area
    # ========================================================================

    # Header
    render_header()

    # Quick start guide (collapsible)
    with st.expander("Quick Start Guide", expanded=False):
        render_quick_guide()

    render_divider()

    # Two-column layout
    col_input, col_result = st.columns([1, 1], gap="large")

    # ========================================================================
    # Input Column
    # ========================================================================
    with col_input:
        input_data = render_input_panel(sidebar_config)
        run_flag = render_run_button(sidebar_config, input_data)

    # ========================================================================
    # Result Column
    # ========================================================================
    with col_result:
        render_result_panel(sidebar_config, input_data, run_flag)

    # ========================================================================
    # Footer
    # ========================================================================
    render_footer()


if __name__ == "__main__":
    main()
