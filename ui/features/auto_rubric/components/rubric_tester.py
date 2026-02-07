# -*- coding: utf-8 -*-
"""Test panel component for Auto Rubric feature.

Provides UI for testing generated graders with sample inputs.
Supports both Pointwise (single response scoring) and Listwise (multiple response ranking) modes.
"""

import html
from typing import Any

import streamlit as st
from features.auto_rubric.services.rubric_generator_service import (
    RubricGeneratorService,
)
from shared.i18n import t
from shared.utils.helpers import run_async

from openjudge.graders.llm_grader import LLMGrader


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text)) if text else ""


def render_test_panel(
    grader: LLMGrader | None = None,
    grader_mode: str = "pointwise",
) -> None:
    """Render the test panel for evaluating sample inputs.

    Args:
        grader: The LLMGrader instance to test. If None, shows disabled state.
        grader_mode: The grader mode ("pointwise" or "listwise").
    """
    st.markdown(
        f"""
        <div style="
            font-weight: 600;
            color: #F1F5F9;
            font-size: 1rem;
            margin: 1rem 0 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            <span>🧪</span>
            <span>{t('rubric.test.title')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if grader is None:
        # Disabled state - no grader available
        st.markdown(
            f"""
            <div style="
                background: rgba(30, 41, 59, 0.5);
                border: 1px dashed #475569;
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
            ">
                <div style="color: #64748B; font-size: 0.9rem;">
                    {t('rubric.test.no_grader')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Initialize session state for test results
    if "rubric_test_result" not in st.session_state:
        st.session_state["rubric_test_result"] = None
    if "rubric_test_running" not in st.session_state:
        st.session_state["rubric_test_running"] = False

    # Input fields based on mode
    with st.container():
        test_query = st.text_area(
            t("rubric.test.query"),
            placeholder=t("rubric.test.query_placeholder"),
            height=80,
            key="rubric_test_query",
        )

        if grader_mode == "listwise":
            # Listwise mode: multiple responses for ranking
            _render_listwise_inputs()
            responses = _get_listwise_responses()
            can_run = bool(test_query.strip() and len(responses) >= 2)
        else:
            # Pointwise mode: single response for scoring
            test_response = st.text_area(
                t("rubric.test.response"),
                placeholder=t("rubric.test.response_placeholder"),
                height=120,
                key="rubric_test_response",
            )
            responses = [test_response] if test_response.strip() else []
            can_run = bool(test_query.strip() and test_response.strip())

        # Run button
        run_clicked = st.button(
            f"▶️ {t('rubric.test.run')}",
            type="primary",
            use_container_width=True,
            disabled=not can_run or st.session_state.get("rubric_test_running", False),
            key="rubric_test_run_btn",
        )

        if run_clicked and can_run:
            if grader_mode == "listwise":
                _run_test_listwise(grader, test_query, responses)
            else:
                _run_test_pointwise(grader, test_query, responses[0])

    # Display results
    test_result = st.session_state.get("rubric_test_result")
    if test_result is not None:
        _render_test_result(test_result, grader_mode)


def _render_listwise_inputs() -> None:
    """Render input fields for listwise mode (multiple responses)."""
    st.markdown(
        f"""
        <div style="
            color: #94A3B8;
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        ">{t('rubric.test.responses_hint')}</div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize response count if not exists
    if "rubric_test_response_count" not in st.session_state:
        st.session_state["rubric_test_response_count"] = 2

    response_count = st.session_state["rubric_test_response_count"]

    # Render response inputs
    for i in range(response_count):
        st.text_area(
            f"{t('rubric.test.response')} {i + 1}",
            placeholder=f"{t('rubric.test.response_placeholder')} {i + 1}",
            height=80,
            key=f"rubric_test_response_{i}",
        )

    # Add/remove response buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            f"➕ {t('rubric.test.add_response')}",
            disabled=response_count >= 5,
            key="rubric_test_add_response",
        ):
            st.session_state["rubric_test_response_count"] = response_count + 1
            st.rerun()
    with col2:
        if st.button(
            f"➖ {t('rubric.test.remove_response')}",
            disabled=response_count <= 2,
            key="rubric_test_remove_response",
        ):
            st.session_state["rubric_test_response_count"] = response_count - 1
            # Clean up removed response from session state
            if f"rubric_test_response_{response_count - 1}" in st.session_state:
                del st.session_state[f"rubric_test_response_{response_count - 1}"]
            st.rerun()


def _get_listwise_responses() -> list[str]:
    """Get all non-empty responses for listwise mode."""
    responses = []
    response_count = st.session_state.get("rubric_test_response_count", 2)
    for i in range(response_count):
        response = st.session_state.get(f"rubric_test_response_{i}", "")
        if response.strip():
            responses.append(response.strip())
    return responses


def _run_test_pointwise(grader: LLMGrader, query: str, response: str) -> None:
    """Run pointwise grader test.

    Args:
        grader: The LLMGrader instance.
        query: Test query.
        response: Test response.
    """
    st.session_state["rubric_test_running"] = True
    st.session_state["rubric_test_result"] = None

    with st.spinner(t("rubric.test.running")):
        try:
            service = RubricGeneratorService()
            result = run_async(service.test_grader(grader, query, response))
            st.session_state["rubric_test_result"] = result
        except Exception as e:
            st.session_state["rubric_test_result"] = {
                "success": False,
                "error": str(e),
            }
        finally:
            st.session_state["rubric_test_running"] = False

    st.rerun()


def _run_test_listwise(grader: LLMGrader, query: str, responses: list[str]) -> None:
    """Run listwise grader test.

    Args:
        grader: The LLMGrader instance.
        query: Test query.
        responses: List of responses to rank.
    """
    st.session_state["rubric_test_running"] = True
    st.session_state["rubric_test_result"] = None

    with st.spinner(t("rubric.test.running")):
        try:
            service = RubricGeneratorService()
            result = run_async(service.test_grader_listwise(grader, query, responses))
            st.session_state["rubric_test_result"] = result
        except Exception as e:
            st.session_state["rubric_test_result"] = {
                "success": False,
                "error": str(e),
            }
        finally:
            st.session_state["rubric_test_running"] = False

    st.rerun()


def _render_test_result(result: dict[str, Any], grader_mode: str) -> None:
    """Render the test result.

    Args:
        result: Test result dictionary.
        grader_mode: The grader mode.
    """
    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)

    if not result.get("success", False):
        error = _escape_html(result.get("error", "Unknown error"))
        st.error(f"{t('rubric.test.error')}: {error}")
        return

    # Success result
    if grader_mode == "listwise":
        _render_listwise_result(result)
    else:
        _render_pointwise_result(result)

    # Clear button
    if st.button(
        f"🗑️ {t('rubric.test.clear')}",
        key="rubric_test_clear_btn",
        use_container_width=True,
    ):
        st.session_state["rubric_test_result"] = None
        st.rerun()


def _render_pointwise_result(result: dict[str, Any]) -> None:
    """Render pointwise test result."""
    score = result.get("score")
    reason = result.get("reason", "")

    # Score display
    if score is not None:
        # Determine score color based on value (assuming 0-5 range)
        if isinstance(score, (int, float)):
            if score >= 4:
                score_color = "#22C55E"  # Green
            elif score >= 2:
                score_color = "#F59E0B"  # Amber
            else:
                score_color = "#EF4444"  # Red
        else:
            score_color = "#6366F1"

        st.markdown(
            f"""
            <div style="
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 0.75rem;
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 0.5rem;
                ">
                    <span style="color: #94A3B8; font-size: 0.9rem;">
                        {t('rubric.test.score')}
                    </span>
                    <span style="
                        font-size: 1.5rem;
                        font-weight: 700;
                        color: {score_color};
                    ">{score}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Reason display
    if reason:
        st.markdown(
            f"""
            <div style="
                color: #94A3B8;
                font-size: 0.9rem;
                margin-bottom: 0.5rem;
            ">{t('rubric.test.reason')}</div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area(
            label="reason_display",
            value=reason,
            height=150,
            disabled=True,
            label_visibility="collapsed",
            key="rubric_test_reason_display",
        )


def _render_listwise_result(result: dict[str, Any]) -> None:
    """Render listwise test result."""
    rank = result.get("rank", [])
    reason = result.get("reason", "")

    # Rank display
    if rank:
        st.markdown(
            f"""
            <div style="
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 0.75rem;
            ">
                <div style="
                    color: #94A3B8;
                    font-size: 0.9rem;
                    margin-bottom: 0.5rem;
                ">{t('rubric.test.rank')}</div>
                <div style="
                    display: flex;
                    gap: 0.5rem;
                    flex-wrap: wrap;
                ">
            """,
            unsafe_allow_html=True,
        )

        # Display ranking with badges
        rank_html = ""
        for i, r in enumerate(rank):
            badge_color = "#22C55E" if i == 0 else "#6366F1" if i == 1 else "#94A3B8"
            rank_html += f"""
                <div style="
                    background: {badge_color}20;
                    border: 1px solid {badge_color};
                    border-radius: 4px;
                    padding: 0.25rem 0.75rem;
                    color: {badge_color};
                    font-weight: 600;
                ">#{i + 1}: Response {r}</div>
            """

        st.markdown(
            f"""
                {rank_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Reason display
    if reason:
        st.markdown(
            f"""
            <div style="
                color: #94A3B8;
                font-size: 0.9rem;
                margin-bottom: 0.5rem;
            ">{t('rubric.test.reason')}</div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area(
            label="listwise_reason_display",
            value=reason,
            height=150,
            disabled=True,
            label_visibility="collapsed",
            key="rubric_test_listwise_reason_display",
        )


def render_test_section_compact(
    grader: LLMGrader | None = None,
    grader_mode: str = "pointwise",
) -> None:
    """Render a compact test section within an expander.

    Args:
        grader: The LLMGrader instance to test.
        grader_mode: The grader mode.
    """
    with st.expander(f"🧪 {t('rubric.test.title')}", expanded=False):
        if grader is None:
            st.info(t("rubric.test.no_grader"))
            return

        # Initialize session state
        if "rubric_test_result" not in st.session_state:
            st.session_state["rubric_test_result"] = None

        # Input fields
        test_query = st.text_input(
            t("rubric.test.query"),
            placeholder=t("rubric.test.query_placeholder"),
            key="rubric_test_query_compact",
        )

        if grader_mode == "listwise":
            # Listwise mode: show hint and two response inputs
            st.caption(t("rubric.test.responses_hint"))
            response_1 = st.text_area(
                f"{t('rubric.test.response')} 1",
                height=80,
                key="rubric_test_response_compact_1",
            )
            response_2 = st.text_area(
                f"{t('rubric.test.response')} 2",
                height=80,
                key="rubric_test_response_compact_2",
            )
            responses = [r for r in [response_1, response_2] if r.strip()]
            can_run = bool(test_query.strip() and len(responses) >= 2)

            if st.button(
                f"▶️ {t('rubric.test.run')}",
                disabled=not can_run,
                key="rubric_test_run_compact",
            ):
                with st.spinner(t("rubric.test.running")):
                    try:
                        service = RubricGeneratorService()
                        result = run_async(service.test_grader_listwise(grader, test_query, responses))
                        st.session_state["rubric_test_result"] = result
                    except Exception as e:
                        st.session_state["rubric_test_result"] = {
                            "success": False,
                            "error": str(e),
                        }
        else:
            # Pointwise mode
            test_response = st.text_area(
                t("rubric.test.response"),
                placeholder=t("rubric.test.response_placeholder"),
                height=100,
                key="rubric_test_response_compact",
            )
            can_run = bool(test_query.strip() and test_response.strip())

            if st.button(
                f"▶️ {t('rubric.test.run')}",
                disabled=not can_run,
                key="rubric_test_run_compact",
            ):
                with st.spinner(t("rubric.test.running")):
                    try:
                        service = RubricGeneratorService()
                        result = run_async(service.test_grader(grader, test_query, test_response))
                        st.session_state["rubric_test_result"] = result
                    except Exception as e:
                        st.session_state["rubric_test_result"] = {
                            "success": False,
                            "error": str(e),
                        }

        # Display result
        test_result = st.session_state.get("rubric_test_result")
        if test_result:
            if test_result.get("success"):
                if grader_mode == "listwise":
                    rank = test_result.get("rank", [])
                    reason = test_result.get("reason", "")
                    st.success(f"{t('rubric.test.rank')}: {rank}")
                    if reason:
                        st.text_area(
                            t("rubric.test.reason"),
                            value=reason,
                            height=100,
                            disabled=True,
                            key="compact_listwise_reason_display",
                        )
                else:
                    score = test_result.get("score")
                    reason = test_result.get("reason", "")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric(t("rubric.test.score"), score)
                    with col2:
                        if reason:
                            st.text_area(
                                t("rubric.test.reason"),
                                value=reason,
                                height=100,
                                disabled=True,
                                key="compact_reason_display",
                            )
            else:
                st.error(test_result.get("error", "Unknown error"))
