# -*- coding: utf-8 -*-
"""Logo and branding components for OpenJudge Studio."""

import base64
from pathlib import Path

import streamlit as st
from shared.constants import APP_NAME, APP_VERSION

# Logo path
LOGO_PATH = Path(__file__).parent.parent.parent / "assets" / "logo.svg"


def render_logo_and_title() -> None:
    """Render logo and title section (single line compact layout)."""
    # Read logo as base64 for inline embedding
    logo_data = ""
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()

    if logo_data:
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_data}" style="width: 36px; height: 36px;" />'
    else:
        logo_html = """<div style="width: 36px; height: 36px;
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            border-radius: 8px; display: flex; align-items: center;
            justify-content: center; font-size: 1rem; color: white;
            font-weight: 700;">OJ</div>"""

    st.markdown(
        f"""<div class="sidebar-header">
            {logo_html}
            <div class="sidebar-header-text">
                <div style="font-size: 1rem; font-weight: 700; color: #F1F5F9; line-height: 1.2;">
                    {APP_NAME}
                </div>
                <div style="font-size: 0.65rem; color: #64748B;">
                    v{APP_VERSION}
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_feature_header(icon: str, name: str, description: str) -> None:
    """Render a feature header with icon and description.

    Args:
        icon: Emoji icon
        name: Feature name
        description: Feature description
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
                <span>{icon}</span>
                <span>{name}</span>
            </h1>
            <p style="color: #94A3B8; margin: 0.25rem 0 0 0; font-size: 0.95rem;">
                {description}
            </p>
        </div>""",
        unsafe_allow_html=True,
    )
