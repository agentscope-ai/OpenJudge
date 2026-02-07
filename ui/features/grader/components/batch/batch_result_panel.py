# -*- coding: utf-8 -*-
"""Result panel for batch evaluation.

Displays:
- Summary statistics
- Score distribution
- Results table with filtering
- Export functionality
"""

from typing import Any

import streamlit as st
from features.grader.services.batch_history_manager import BatchHistoryManager
from shared.components.common import render_section_header
from shared.styles.theme import get_score_color


def _render_summary_cards(summary: dict[str, Any], score_range: tuple[float, float]) -> None:
    """Render summary statistics cards."""
    avg_score = summary.get("avg_score")
    pass_rate = summary.get("pass_rate")
    total_count = summary.get("total_count", 0)
    success_count = summary.get("success_count", 0)
    failed_count = summary.get("failed_count", 0)
    passed_count = summary.get("passed_count", 0)

    col1, col2, col3 = st.columns(3)

    with col1:
        if avg_score is not None:
            score_color = get_score_color(avg_score, score_range[1])
            score_display = f"{avg_score:.2f}"
        else:
            score_color = "#64748B"
            score_display = "--"

        st.markdown(
            f"""<div style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 1.25rem;
                text-align: center;
            ">
                <div style="font-size: 2rem; font-weight: 700; color: {score_color};">
                    {score_display}
                </div>
                <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase;">
                    / {score_range[1]}
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.5rem;">
                    Avg Score / 平均分
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        if pass_rate is not None:
            rate_pct = pass_rate * 100
            rate_color = "#10B981" if rate_pct >= 80 else "#F59E0B" if rate_pct >= 50 else "#EF4444"
            rate_display = f"{rate_pct:.1f}%"
        else:
            rate_color = "#64748B"
            rate_display = "--"

        st.markdown(
            f"""<div style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 1.25rem;
                text-align: center;
            ">
                <div style="font-size: 2rem; font-weight: 700; color: {rate_color};">
                    {rate_display}
                </div>
                <div style="font-size: 0.75rem; color: #64748B;">
                    {passed_count} / {success_count}
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.5rem;">
                    Pass Rate / 通过率
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""<div style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 1.25rem;
                text-align: center;
            ">
                <div style="font-size: 1.5rem; font-weight: 700;">
                    <span style="color: #10B981;">{success_count}</span>
                    <span style="color: #64748B;"> / </span>
                    <span style="color: #EF4444;">{failed_count}</span>
                </div>
                <div style="font-size: 0.75rem; color: #64748B;">
                    of {total_count} total
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.5rem;">
                    Success / Failed
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_score_distribution(results: list[dict[str, Any]], score_range: tuple[float, float]) -> None:
    """Render score distribution chart."""
    # Extract scores
    scores = [r.get("score") for r in results if r.get("status") == "success" and r.get("score") is not None]

    if not scores:
        return

    # Create histogram data
    min_score, max_score = score_range
    num_bins = 10 if max_score <= 1 else 5

    if max_score <= 1:
        bins = [i / num_bins for i in range(num_bins + 1)]
    else:
        bins = list(range(int(min_score), int(max_score) + 2))

    # Count scores in each bin
    counts = [0] * (len(bins) - 1)
    for score in scores:
        for i in range(len(bins) - 1):
            if bins[i] <= score < bins[i + 1]:
                counts[i] += 1
                break
        else:
            # Handle max value
            if score == bins[-1]:
                counts[-1] += 1

    # Create bar chart using HTML/CSS
    max_count = max(counts) if counts else 1
    bar_width = 100 / len(counts)

    bars_html = ""
    for i, count in enumerate(counts):
        height_pct = (count / max_count * 100) if max_count > 0 else 0
        label = f"{bins[i]:.1f}" if max_score <= 1 else f"{int(bins[i])}"

        bars_html += f"""<br>
            <div style="
                display: flex;
                flex-direction: column;
                align-items: center;
                width: {bar_width}%;
            ">
                <div style="
                    height: 60px;
                    width: 80%;
                    display: flex;
                    align-items: flex-end;
                ">
                    <div style="
                        width: 100%;
                        height: {height_pct}%;
                        background: linear-gradient(180deg, #6366F1, #4F46E5);
                        border-radius: 4px 4px 0 0;
                        min-height: {2 if count > 0 else 0}px;
                    "></div>
                </div>
                <div style="font-size: 0.65rem; color: #64748B; margin-top: 0.25rem;">
                    {label}
                </div>
                <div style="font-size: 0.6rem; color: #94A3B8;">
                    {count}
                </div>
            </div>"""

    st.markdown(
        f"""<div style="margin: 1rem 0;">
            <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.5rem;">
                Score Distribution / 分数分布
            </div>
            <div style="
                display: flex;
                justify-content: space-around;
                align-items: flex-end;
                padding: 0.5rem;
                background: rgba(30, 41, 59, 0.3);
                border-radius: 8px;
            ">
                {bars_html}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_results_table(
    results: list[dict[str, Any]],
    score_range: tuple[float, float],
    task_id: str,
    page_size: int = 20,
) -> None:
    """Render paginated results table."""
    if not results:
        return

    # Filter options
    col_filter, col_sort, col_page = st.columns([2, 2, 1])

    with col_filter:
        filter_option = st.selectbox(
            "Filter / 筛选",
            options=["All", "Passed", "Failed", "Errors"],
            key=f"batch_result_filter_{task_id}",
            label_visibility="collapsed",
        )

    with col_sort:
        sort_option = st.selectbox(
            "Sort / 排序",
            options=["Index ↑", "Index ↓", "Score ↑", "Score ↓"],
            key=f"batch_result_sort_{task_id}",
            label_visibility="collapsed",
        )

    # Apply filters
    filtered_results = results
    if filter_option == "Passed":
        filtered_results = [r for r in results if r.get("passed") is True]
    elif filter_option == "Failed":
        filtered_results = [r for r in results if r.get("passed") is False and r.get("status") == "success"]
    elif filter_option == "Errors":
        filtered_results = [r for r in results if r.get("status") == "error"]

    # Apply sorting
    if sort_option == "Index ↑":
        filtered_results = sorted(filtered_results, key=lambda x: x.get("index", 0))
    elif sort_option == "Index ↓":
        filtered_results = sorted(filtered_results, key=lambda x: x.get("index", 0), reverse=True)
    elif sort_option == "Score ↑":
        filtered_results = sorted(filtered_results, key=lambda x: x.get("score") or 0)
    elif sort_option == "Score ↓":
        filtered_results = sorted(filtered_results, key=lambda x: x.get("score") or 0, reverse=True)

    # Pagination
    total_pages = (len(filtered_results) + page_size - 1) // page_size
    with col_page:
        current_page = st.number_input(
            "Page",
            min_value=1,
            max_value=max(1, total_pages),
            value=1,
            key="batch_result_page",
            label_visibility="collapsed",
        )

    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    page_results = filtered_results[start_idx:end_idx]

    st.markdown(
        f"""<div style="font-size: 0.75rem; color: #64748B; margin-bottom: 0.5rem;">
            Showing {start_idx + 1}-{min(end_idx, len(filtered_results))} of {len(filtered_results)} results
            (Page {current_page}/{total_pages})
        </div>""",
        unsafe_allow_html=True,
    )

    # Render table
    for result in page_results:
        index = result.get("index", 0)
        status = result.get("status", "unknown")
        score = result.get("score")
        passed = result.get("passed")
        reason = result.get("reason", "")
        error = result.get("error", "")

        # Status styling
        if status == "error":
            status_bg = "rgba(239, 68, 68, 0.1)"
            status_border = "#EF4444"
            status_icon = "❌"
        elif passed:
            status_bg = "rgba(16, 185, 129, 0.1)"
            status_border = "#10B981"
            status_icon = "✓"
        else:
            status_bg = "rgba(245, 158, 11, 0.1)"
            status_border = "#F59E0B"
            status_icon = "✗"

        score_display = f"{score:.2f}" if score is not None else "--"
        score_color = get_score_color(score, score_range[1]) if score is not None else "#64748B"

        with st.container():
            st.markdown(
                f"""<div style="
                    background: {status_bg};
                    border-left: 3px solid {status_border};
                    border-radius: 0 8px 8px 0;
                    padding: 0.75rem 1rem;
                    margin-bottom: 0.5rem;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                            <span style="
                                font-weight: 600;
                                color: #F1F5F9;
                                min-width: 3rem;
                            ">#{index + 1}</span>
                            <span style="color: {status_border}; font-size: 1rem;">{status_icon}</span>
                        </div>
                        <div style="
                            font-size: 1.25rem;
                            font-weight: 700;
                            color: {score_color};
                        ">{score_display}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Expandable details
            with st.expander(f"Details for #{index + 1}", expanded=False):
                if error:
                    st.error(f"**Error:** {error}")
                if reason:
                    st.markdown(f"**Reason:** {reason}")

                input_data = result.get("input", {})
                if input_data:
                    st.markdown("**Input Data:**")
                    st.json(input_data)


def _render_export_buttons(task_id: str, history_manager: BatchHistoryManager) -> None:
    """Render export buttons."""
    col1, col2 = st.columns(2)

    with col1:
        json_data = history_manager.export_results(task_id, "json")
        if json_data:
            st.download_button(
                label="📥 Export JSON",
                data=json_data,
                file_name=f"{task_id}_results.json",
                mime="application/json",
                use_container_width=True,
                key=f"download_json_{task_id}",
            )

    with col2:
        csv_data = history_manager.export_results(task_id, "csv")
        if csv_data:
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"{task_id}_results.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"download_csv_{task_id}",
            )


def render_batch_result_panel(
    task_id: str,
    results: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
    score_range: tuple[float, float] = (0, 1),
    history_manager: BatchHistoryManager | None = None,
) -> None:
    """Render the batch result panel.

    Args:
        task_id: Task ID for export
        results: List of evaluation results
        summary: Summary statistics
        score_range: Score range tuple (min, max)
        history_manager: History manager for export
    """
    render_section_header("Evaluation Results / 评估结果")

    if not results and not summary:
        st.info("No results yet. Run an evaluation to see results here.")
        return

    # Load from history if not provided
    if history_manager is None:
        history_manager = BatchHistoryManager()

    if results is None:
        results = history_manager.get_task_results(task_id) or []

    if summary is None:
        details = history_manager.get_task_details(task_id)
        summary = details.get("summary", {}) if details else {}

    # Summary cards
    _render_summary_cards(summary, score_range)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Score distribution
    if results:
        _render_score_distribution(results, score_range)

    # Export buttons
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    _render_export_buttons(task_id, history_manager)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Results table
    if results:
        st.markdown(
            """<div style="font-weight: 500; color: #94A3B8; margin-bottom: 0.5rem;">
                Detailed Results / 详细结果
            </div>""",
            unsafe_allow_html=True,
        )
        _render_results_table(results, score_range, task_id)


def render_empty_result_state() -> None:
    """Render empty result state."""
    render_section_header("Evaluation Results / 评估结果")

    st.markdown(
        """<div style="
            text-align: center;
            padding: 2rem;
            color: #64748B;
            background: rgba(30, 41, 59, 0.3);
            border-radius: 8px;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-size: 0.9rem;">
                Results will appear here after evaluation completes<br/>
                评估完成后结果将显示在此处
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
