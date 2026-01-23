# -*- coding: utf-8 -*-
"""Grader feature implementation for OpenJudge Studio."""

from typing import Any

import streamlit as st
from core.base_feature import BaseFeature
from features.grader.components.input_panel import render_input_panel_with_button
from features.grader.components.result_panel import render_result_panel
from features.grader.components.sidebar import render_grader_sidebar
from shared.components.common import render_divider


class GraderFeature(BaseFeature):
    """Grader evaluation feature.

    Provides UI for evaluating LLM responses using OpenJudge's built-in graders.
    Supports multiple grader categories including common, text, format, code,
    math, multimodal, and agent graders.
    """

    feature_id = "grader"
    feature_name = "Grader 评估"
    feature_icon = "⚖️"
    feature_description = "使用内置 Grader 对单条数据进行评估"
    order = 1

    def render_sidebar(self) -> dict[str, Any]:
        """Render the grader-specific sidebar configuration.

        Returns:
            Dictionary containing all sidebar configuration values
        """
        return render_grader_sidebar()

    def render_main_content(self, sidebar_config: dict[str, Any]) -> None:
        """Render the main content area for grader evaluation.

        Args:
            sidebar_config: Configuration from the sidebar
        """
        # Quick start guide (collapsible)
        with st.expander("Quick Start Guide", expanded=False):
            self._render_quick_guide()

        render_divider()

        # Two-column layout
        col_input, col_result = st.columns([1, 1], gap="large")

        # Input Column
        with col_input:
            input_data, run_flag = render_input_panel_with_button(sidebar_config)

        # Result Column
        with col_result:
            render_result_panel(sidebar_config, input_data, run_flag)

    def _render_quick_guide(self) -> None:
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

    def on_mount(self) -> None:
        """Initialize grader feature state when mounted."""
        # Initialize evaluation result if not present
        if "evaluation_result" not in st.session_state:
            st.session_state.evaluation_result = None
