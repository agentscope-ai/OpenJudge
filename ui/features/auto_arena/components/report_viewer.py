# -*- coding: utf-8 -*-
"""Report viewer component for Auto Arena feature.

Displays evaluation reports with markdown rendering and export options.
"""

from pathlib import Path
from typing import Any

import streamlit as st
from features.auto_arena.services.history_manager import HistoryManager


def _render_comparison_detail(detail: dict[str, Any], index: int) -> None:
    """Render a single comparison detail.

    Args:
        detail: Comparison detail dict
        index: Index for unique keys
    """
    winner = detail.get("winner", "")
    score = detail.get("score", 0.5)
    model_a = detail.get("model_a", "Model A")
    model_b = detail.get("model_b", "Model B")

    # Determine winner display
    if winner == "model_a":
        winner_name = model_a
        winner_color = "#10B981"
    elif winner == "model_b":
        winner_name = model_b
        winner_color = "#6366F1"
    else:
        winner_name = "Tie"
        winner_color = "#94A3B8"

    with st.expander(f"Query {index + 1}: {detail.get('query', '')[:50]}...", expanded=False):
        st.markdown(
            f"""<div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.5rem;">
                Winner: <span style="color: {winner_color}; font-weight: 600;">{winner_name}</span>
                (Score: {score:.2f})
            </div>""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{model_a}**")
            st.text_area(
                f"Response A ({model_a})",
                value=detail.get("response_a", ""),
                height=150,
                key=f"detail_resp_a_{index}",
                label_visibility="collapsed",
                disabled=True,
            )
        with col2:
            st.markdown(f"**{model_b}**")
            st.text_area(
                f"Response B ({model_b})",
                value=detail.get("response_b", ""),
                height=150,
                key=f"detail_resp_b_{index}",
                label_visibility="collapsed",
                disabled=True,
            )

        if detail.get("reason"):
            st.markdown("**Reason:**")
            st.markdown(f"_{detail['reason']}_")


def render_report_viewer(
    task_id: str | None = None,
    task_details: dict[str, Any] | None = None,
    on_back: Any = None,
) -> None:
    """Render the report viewer for a specific task.

    Args:
        task_id: ID of the task to view
        task_details: Pre-loaded task details (optional)
        on_back: Callback when back button is clicked
    """
    history_manager = HistoryManager()

    # Load details if not provided
    if task_details is None and task_id:
        task_details = history_manager.get_task_details(task_id)

    if not task_details:
        st.warning("Task not found / 任务未找到")
        if on_back and st.button("← Back"):
            on_back()
        return

    # Header with back button
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if on_back and st.button("← Back"):
            on_back()
    with col_title:
        st.markdown(
            f"""<div style="font-size: 1.25rem; font-weight: 600; color: #F1F5F9;">
                📊 Evaluation Report
            </div>
            <div style="font-size: 0.8rem; color: #64748B;">
                Task ID: {task_details.get('task_id', 'Unknown')}
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="custom-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

    # Tabs for different views
    tab_overview, tab_report, tab_details, tab_export = st.tabs(["📈 Overview", "📄 Report", "🔍 Details", "💾 Export"])

    results = task_details.get("results", {})
    result = results.get("result", {})

    with tab_overview:
        _render_overview_tab(result, task_details)

    with tab_report:
        _render_report_tab(task_id, history_manager)

    with tab_details:
        _render_details_tab(task_details)

    with tab_export:
        _render_export_tab(task_id, task_details, history_manager)


def _render_overview_tab(result: dict[str, Any], task_details: dict[str, Any]) -> None:
    """Render the overview tab."""
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Queries", result.get("total_queries", 0))
    with col2:
        st.metric("Comparisons", result.get("total_comparisons", 0))
    with col3:
        st.metric("Best Model", result.get("best_pipeline", "N/A"))

    st.markdown('<div class="custom-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

    # Rankings
    st.markdown("### Model Rankings")
    rankings = result.get("rankings", [])
    for rank, (name, win_rate) in enumerate(rankings, 1):
        bar_width = int(win_rate * 100)
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

        st.markdown(
            f"""<div style="
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.75rem;
                background: rgba(30, 41, 59, 0.3);
                border-radius: 8px;
                margin-bottom: 0.5rem;
            ">
                <span style="font-size: 1.5rem; width: 2rem; text-align: center;">{medal}</span>
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: #F1F5F9;">{name}</div>
                    <div style="
                        background: rgba(100, 116, 139, 0.3);
                        border-radius: 4px;
                        height: 6px;
                        margin-top: 0.25rem;
                    ">
                        <div style="
                            background: linear-gradient(90deg, #6366F1, #8B5CF6);
                            width: {bar_width}%;
                            height: 100%;
                            border-radius: 4px;
                        "></div>
                    </div>
                </div>
                <span style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9; min-width: 4rem; text-align: right;">
                    {win_rate:.1%}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Chart
    chart_path = task_details.get("task_dir")
    if chart_path:
        chart_file = Path(chart_path) / "win_rate_chart.png"
        if chart_file.exists():
            st.markdown("### Win Rate Chart")
            st.image(str(chart_file), use_container_width=True)


def _render_report_tab(task_id: str | None, history_manager: HistoryManager) -> None:
    """Render the report tab with markdown content."""
    if not task_id:
        st.info("Report not available")
        return

    report_content = history_manager.get_report_content(task_id)
    if report_content:
        st.markdown(report_content)
    else:
        st.info("Report not generated for this evaluation / 此评估未生成报告")


def _render_details_tab(task_details: dict[str, Any]) -> None:
    """Render the comparison details tab."""
    comparison_details = task_details.get("comparison_details", [])

    if not comparison_details:
        st.info("No comparison details available / 无比较详情")
        return

    st.markdown(f"**{len(comparison_details)} comparisons** (showing unique queries)")

    # Filter to show only original order comparisons (to avoid duplicates)
    unique_details = [d for d in comparison_details if d.get("order") == "original"]

    for i, detail in enumerate(unique_details[:20]):  # Limit to 20
        _render_comparison_detail(detail, i)

    if len(unique_details) > 20:
        st.info(f"Showing 20 of {len(unique_details)} comparisons")


def _render_export_tab(
    task_id: str | None,
    task_details: dict[str, Any],
    history_manager: HistoryManager,
) -> None:
    """Render the export tab."""
    st.markdown("### Export Options")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**JSON Export**")
        st.markdown(
            '<div style="font-size: 0.8rem; color: #94A3B8;">'
            "Full evaluation data including queries, responses, and results</div>",
            unsafe_allow_html=True,
        )
        if task_id:
            json_data = history_manager.export_task(task_id, "json")
            if json_data:
                st.download_button(
                    "📥 Download JSON",
                    data=json_data,
                    file_name=f"evaluation_{task_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    with col2:
        st.markdown("**CSV Export**")
        st.markdown(
            '<div style="font-size: 0.8rem; color: #94A3B8;">Rankings and win rates in spreadsheet format</div>',
            unsafe_allow_html=True,
        )
        if task_id:
            csv_data = history_manager.export_task(task_id, "csv")
            if csv_data:
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name=f"evaluation_{task_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # Report download
    st.markdown('<div class="custom-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
    st.markdown("**Report Download**")

    task_dir = task_details.get("task_dir")
    if task_dir:
        report_path = Path(task_dir) / "evaluation_report.md"
        chart_path = Path(task_dir) / "win_rate_chart.png"

        col3, col4 = st.columns(2)
        with col3:
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        "📄 Download Report (MD)",
                        data=f.read(),
                        file_name="evaluation_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
            else:
                st.button("📄 Report N/A", disabled=True, use_container_width=True)

        with col4:
            if chart_path.exists():
                with open(chart_path, "rb") as f:
                    st.download_button(
                        "📊 Download Chart (PNG)",
                        data=f.read(),
                        file_name="win_rate_chart.png",
                        mime="image/png",
                        use_container_width=True,
                    )
            else:
                st.button("📊 Chart N/A", disabled=True, use_container_width=True)
