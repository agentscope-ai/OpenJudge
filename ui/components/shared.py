# -*- coding: utf-8 -*-
"""Shared UI components for OpenJudge Studio."""

import streamlit as st
from config.constants import APP_NAME, APP_VERSION


def render_header() -> None:
    """Render the main header with title and subtitle."""
    st.markdown(
        f'<p class="main-header">{APP_NAME}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Evaluate LLM responses with precision and insight</p>',
        unsafe_allow_html=True,
    )


def render_quick_guide() -> None:
    """Render the quick start guide."""
    st.markdown(
        """
        <div class="feature-card">
            <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem;">
                Quick Start Guide
            </div>
            <div class="guide-step">
                <div class="guide-number">1</div>
                <div class="guide-text">Configure your API endpoint and key in the sidebar</div>
            </div>
            <div class="guide-step">
                <div class="guide-number">2</div>
                <div class="guide-text">Select a grader category and specific grader</div>
            </div>
            <div class="guide-step">
                <div class="guide-number">3</div>
                <div class="guide-text">Enter your evaluation data (query, response, etc.)</div>
            </div>
            <div class="guide-step">
                <div class="guide-number">4</div>
                <div class="guide-text">Click "Run Evaluation" to see results</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the footer."""
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="footer">
            <div style="margin-bottom: 0.5rem;">
                Built with <strong>OpenJudge Framework</strong> and <strong>Streamlit</strong>
            </div>
            <div style="font-size: 0.7rem;">
                © 2024 OpenJudge • v{APP_VERSION} •
                <a href="https://github.com/modelscope/OpenJudge">GitHub</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_divider() -> None:
    """Render a styled divider."""
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


def render_section_header(title: str, icon: str = "") -> None:
    """Render a section header.

    Args:
        title: Section title
        icon: Optional emoji icon
    """
    icon_part = f"{icon} " if icon else ""
    st.markdown(
        f'<div class="section-header">{icon_part}{title}</div>',
        unsafe_allow_html=True,
    )


def render_category_header(title: str) -> None:
    """Render a category header (smaller, uppercase).

    Args:
        title: Category title
    """
    st.markdown(
        f'<div class="category-header">{title}</div>',
        unsafe_allow_html=True,
    )


def render_info_card(content: str) -> None:
    """Render an info card.

    Args:
        content: Card content (HTML supported)
    """
    st.markdown(
        f"""
        <div class="info-card">
            <div style="font-size: 0.85rem; color: #94A3B8;">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Render empty state placeholder."""
    st.markdown(
        """
        <div class="empty-state animate-fade-in">
            <div class="empty-state-icon animate-pulse" style="
                width: 64px;
                height: 64px;
                margin: 0 auto 1rem;
                background: linear-gradient(135deg, #334155, #475569);
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2">
                    <path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>
                </svg>
            </div>
            <div style="font-size: 1.1rem; font-weight: 500; color: #94A3B8; margin-bottom: 0.5rem;">
                Ready to Evaluate
            </div>
            <div style="color: #64748B;">
                Enter your data and click <strong>Run Evaluation</strong> to see results
            </div>
            <div style="margin-top: 1rem; font-size: 0.875rem; color: #475569;">
                Tip: Use the "Load Example" button to try with sample data
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
