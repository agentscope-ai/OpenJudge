# -*- coding: utf-8 -*-
"""Common shared UI components for OpenJudge Studio."""

import streamlit as st
from shared.constants import APP_VERSION
from shared.i18n import t


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
                {t("app.footer", version=APP_VERSION)} •
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


def render_empty_state(
    title: str = "Ready to Evaluate",
    description: str = "Enter your data and click <strong>Run Evaluation</strong> to see results",
    tip: str = "",
) -> None:
    """Render empty state placeholder.

    Args:
        title: Main title text
        description: Description text (HTML supported)
        tip: Optional tip text
    """
    tip_html = ""
    if tip:
        tip_html = f'<div style="margin-top: 1rem; font-size: 0.875rem; color: #94A3B8;">{tip}</div>'

    st.markdown(
        f"""
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
                {title}
            </div>
            <div style="color: #64748B;">
                {description}
            </div>
            {tip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_bar(progress: float, color: str = "#6366F1") -> None:
    """Render a custom progress bar.

    Args:
        progress: Progress value between 0 and 1
        color: Bar color (hex)
    """
    pct = min(max(progress * 100, 0), 100)
    st.markdown(
        f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {pct}%; background: {color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(value: str, label: str, color: str = "#F1F5F9") -> None:
    """Render a metric card.

    Args:
        value: Metric value to display
        label: Metric label
        color: Value color (hex)
    """
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color};">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str, text: str) -> None:
    """Render a status badge.

    Args:
        status: Status type ('pass', 'fail', 'warning')
        text: Badge text
    """
    status_class = f"status-{status}" if status in ["pass", "fail", "warning"] else "status-warning"
    st.markdown(
        f"""
        <span class="status-badge {status_class}">{text}</span>
        """,
        unsafe_allow_html=True,
    )
