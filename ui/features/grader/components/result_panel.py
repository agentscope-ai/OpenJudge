# -*- coding: utf-8 -*-
"""Result panel component for Grader feature."""

import html
import logging
import time
from typing import Any

import streamlit as st
from features.grader.config.grader_registry import GRADER_REGISTRY
from features.grader.services.grader_factory import (
    create_grader,
    run_agent_evaluation,
    run_evaluation,
    run_multimodal_evaluation,
)
from features.grader.services.single_evaluation_logger import log_single_evaluation
from shared.components.common import render_empty_state, render_section_header
from shared.i18n import t
from shared.services.model_factory import create_model
from shared.styles.theme import get_score_color
from shared.utils.helpers import (
    format_elapsed_time,
    format_score_display,
    parse_json_safely,
)

from openjudge.graders.schema import GraderError


def _render_score_card(
    score: float,
    threshold: float,
    grader_name: str,
    reason: str,
    score_range: tuple[float, float] = (1, 5),
) -> None:
    """Render the score display card.

    Args:
        score: Evaluation score
        threshold: Pass/fail threshold
        grader_name: Name of the grader used
        reason: Evaluation reason/explanation
        score_range: (min, max) score range
    """
    # Format score display
    max_score = score_range[1]
    score_text, range_label = format_score_display(score, score_range)
    score_color = get_score_color(score, max_score)

    # Determine pass/fail
    is_pass = score >= threshold
    status_text = t("grader.result.passed") if is_pass else t("grader.result.failed_status")
    status_icon = "✓" if is_pass else "✗"
    status_bg = "rgba(34, 197, 94, 0.2)" if is_pass else "rgba(239, 68, 68, 0.2)"
    status_border = "#22C55E" if is_pass else "#EF4444"
    status_color = "#22C55E" if is_pass else "#EF4444"

    # Calculate progress percentage
    progress_pct = (score - score_range[0]) / (score_range[1] - score_range[0]) * 100

    # Score display
    st.markdown(
        f"""<div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        ">
            <div style="font-size: 3.5rem; font-weight: 700; color: {score_color}; margin-bottom: 0.25rem;">
                {score_text}
            </div>
            <div style="font-size: 0.875rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em;">
                {range_label}
            </div>
            <div style="margin: 1.5rem 0; height: 8px; background: #1E293B; border-radius: 4px; overflow: hidden;">
                <div style="width: {progress_pct}%; height: 100%; background: {score_color}; border-radius: 4px;"></div>
            </div>
            <div style="margin-top: 1.5rem;">
                <span style="
                    display: inline-block;
                    padding: 0.5rem 1.5rem;
                    background: {status_bg};
                    border: 1px solid {status_border};
                    border-radius: 20px;
                    font-weight: 600;
                    font-size: 0.875rem;
                    color: {status_color};
                ">{status_icon} {status_text}</span>
            </div>
            <div style="margin-top: 1rem; color: #64748B; font-size: 0.8rem;">
                {t("grader.result.threshold_grader", threshold=threshold, grader=grader_name)}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Reason card
    st.markdown(
        f"""<div style="
            margin-top: 1rem;
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.25rem;
        ">
            <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem; font-size: 0.95rem;">
                {t("grader.result.evaluation_reason")}
            </div>
            <div style="color: #CBD5E1; line-height: 1.7; font-size: 0.9rem; white-space: pre-wrap;">
                {reason}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_error_state(error: GraderError) -> None:
    """Render error state with troubleshooting tips.

    Args:
        error: GraderError object
    """
    # Escape HTML special characters to prevent XSS and HTML structure breaking
    escaped_error = html.escape(str(error.error))
    escaped_reason = html.escape(str(error.reason))

    st.markdown(
        f"""
        <div style="
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                     stroke="#EF4444" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="15" y1="9" x2="9" y2="15"/>
                    <line x1="9" y1="9" x2="15" y2="15"/>
                </svg>
                <span style="font-size: 1.1rem; font-weight: 600; color: #EF4444;">
                    {t("grader.result.evaluation_failed")}
                </span>
            </div>
            <div style="color: #FCA5A5; margin-bottom: 1rem;">
                {escaped_error}
            </div>
            <div style="
                background: #1E293B;
                border-radius: 8px;
                padding: 1rem;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.8rem;
                color: #94A3B8;
                overflow-x: auto;
                white-space: pre-wrap;
            ">
                {escaped_reason}
            </div>
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(239, 68, 68, 0.2);">
                <div style="font-weight: 500; color: #F1F5F9; margin-bottom: 0.5rem;">
                    {t("grader.result.troubleshooting")}
                </div>
                <ul style="color: #94A3B8; font-size: 0.875rem; margin: 0; padding-left: 1.25rem;">
                    <li>{t("grader.result.tip_api_key")}</li>
                    <li>{t("grader.result.tip_endpoint")}</li>
                    <li>{t("grader.result.tip_model")}</li>
                    <li>{t("grader.result.tip_vision")}</li>
                    <li>{t("grader.result.tip_format")}</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_evaluation(
    sidebar_config: dict[str, Any],
    input_data: dict[str, Any],
    grader_name: str,
    grader_config: dict[str, Any],
) -> None:
    """Run evaluation and save results to session state.

    Args:
        sidebar_config: Configuration from sidebar
        input_data: Input data from input panel
        grader_name: Name of the grader
        grader_config: Grader configuration
    """
    with st.status(t("grader.result.running"), expanded=True) as status:
        try:
            start_time = time.time()

            # Create model if required
            model = None
            if grader_config.get("requires_model", False):
                st.write(t("grader.result.creating_model"))
                model = create_model(
                    api_key=sidebar_config.get("api_key", ""),
                    base_url=sidebar_config.get("api_endpoint"),
                    model_name=sidebar_config.get("model_name", ""),
                )

            # Create grader
            st.write(t("grader.result.initializing", grader=grader_name))
            grader = create_grader(
                grader_name=grader_name,
                model=model,
                threshold=sidebar_config.get("threshold", 0.5),
                language=sidebar_config.get("language"),
                **sidebar_config.get("extra_params", {}),
            )

            # Run evaluation
            st.write("Running evaluation...")
            result = _execute_evaluation(grader, grader_config, input_data)
            elapsed = time.time() - start_time

            # Save to session state
            st.session_state.evaluation_result = result
            st.session_state.last_grader = grader_name
            st.session_state.last_threshold = sidebar_config.get("threshold", 0.5)
            st.session_state.elapsed_time = elapsed

            # Log evaluation for analytics
            log_single_evaluation(
                grader_name=grader_name,
                input_data=input_data,
                result=result,
                threshold=sidebar_config.get("threshold", 0.5),
                elapsed_time=elapsed,
                model_name=sidebar_config.get("model_name"),
                extra_params=sidebar_config.get("extra_params"),
            )

            status.update(
                label=f"Evaluation complete ({format_elapsed_time(elapsed)})",
                state="complete",
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.exception("Evaluation failed with an unexpected error.")
            status.update(label="Evaluation failed", state="error")
            error_result = GraderError(
                name=grader_name,
                error=str(e),
                reason=f"Exception: {type(e).__name__}: {str(e)}",
            )
            st.session_state.evaluation_result = error_result

            # Log failed evaluation for analytics
            log_single_evaluation(
                grader_name=grader_name,
                input_data=input_data,
                result=error_result,
                threshold=sidebar_config.get("threshold", 0.5),
                elapsed_time=time.time() - start_time,
                model_name=sidebar_config.get("model_name"),
                extra_params=sidebar_config.get("extra_params"),
            )


def _execute_evaluation(grader, grader_config: dict[str, Any], input_data: dict[str, Any]):
    """Execute evaluation based on grader type.

    Args:
        grader: Grader instance
        grader_config: Grader configuration
        input_data: Input data

    Returns:
        Evaluation result
    """
    category = grader_config.get("category", "common")

    if category == "multimodal":
        return run_multimodal_evaluation(
            grader=grader,
            response_content=input_data.get("response", []),
            query=input_data.get("query", ""),
        )

    if category == "agent":
        tool_defs = parse_json_safely(input_data.get("tool_definitions", ""), default=[])
        tool_calls = parse_json_safely(input_data.get("tool_calls", ""), default=[])
        ref_tool_calls = parse_json_safely(input_data.get("reference_tool_calls", ""), default=None)
        return run_agent_evaluation(
            grader=grader,
            query=input_data.get("query", ""),
            tool_definitions=tool_defs,
            tool_calls=tool_calls,
            reference_tool_calls=ref_tool_calls,
        )

    return run_evaluation(
        grader=grader,
        query=input_data.get("query", ""),
        response=input_data.get("response", ""),
        reference_response=input_data.get("reference_response", ""),
        context=input_data.get("context", ""),
    )


def _display_results(grader_name: str) -> None:
    """Display evaluation results.

    Args:
        grader_name: Current grader name
    """
    result = st.session_state.evaluation_result

    if not result:
        render_empty_state(
            title="Ready to Evaluate",
            description="Enter your data and click <strong>Run Evaluation</strong> to see results",
            tip='Tip: Use the "Load Example" button to try with sample data',
        )
        return

    if isinstance(result, GraderError):
        _render_error_state(result)
        return

    last_grader = st.session_state.get("last_grader", grader_name)
    last_config = GRADER_REGISTRY.get(last_grader, {})
    score_range = last_config.get("score_range", (0, 1))

    _render_score_card(
        score=result.score,
        threshold=st.session_state.get("last_threshold", 0.5),
        grader_name=last_grader,
        reason=result.reason,
        score_range=score_range,
    )

    if result.metadata:
        with st.expander("View Metadata"):
            st.json(result.metadata)

    if "elapsed_time" in st.session_state:
        st.caption(f"Completed in {format_elapsed_time(st.session_state.elapsed_time)}")


def render_result_panel(
    sidebar_config: dict[str, Any],
    input_data: dict[str, Any],
    run_evaluation_flag: bool,
) -> None:
    """Render the result panel.

    Args:
        sidebar_config: Configuration from sidebar
        input_data: Input data from input panel
        run_evaluation_flag: Whether to run evaluation
    """
    render_section_header("Evaluation Result")

    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None

    grader_name = sidebar_config.get("grader_name")
    grader_config = sidebar_config.get("grader_config")

    if run_evaluation_flag and grader_name and grader_config:
        _run_evaluation(sidebar_config, input_data, grader_name, grader_config)

    _display_results(grader_name)
