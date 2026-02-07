# -*- coding: utf-8 -*-
"""Result panel for Auto Arena feature.

Displays evaluation results including rankings, win rates, and reports.
"""

from pathlib import Path
from typing import Any

import streamlit as st


def _render_ranking_card(
    rank: int, name: str, win_rate: float, is_best: bool = False  # pylint: disable=unused-argument
) -> None:
    """Render a single ranking card.

    Args:
        rank: Rank position (1-based)
        name: Model name
        win_rate: Win rate (0-1)
        is_best: Whether this is the best model
    """
    # Colors based on rank
    rank_colors = {
        1: {"bg": "rgba(234, 179, 8, 0.1)", "border": "#EAB308", "icon": "🥇"},
        2: {"bg": "rgba(148, 163, 184, 0.1)", "border": "#94A3B8", "icon": "🥈"},
        3: {"bg": "rgba(180, 83, 9, 0.1)", "border": "#B45309", "icon": "🥉"},
    }
    default_color = {"bg": "rgba(100, 116, 139, 0.05)", "border": "#475569", "icon": ""}

    colors = rank_colors.get(rank, default_color)
    bar_width = int(win_rate * 100)

    st.markdown(
        f"""<div style="
            background: {colors['bg']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="
                    font-size: 1.5rem;
                    width: 2.5rem;
                    text-align: center;
                ">{colors['icon'] or f'#{rank}'}</div>
                <div style="flex: 1;">
                    <div style="
                        font-weight: 600;
                        color: #F1F5F9;
                        font-size: 1rem;
                    ">{name}</div>
                    <div style="
                        background: rgba(100, 116, 139, 0.3);
                        border-radius: 4px;
                        height: 8px;
                        margin-top: 0.5rem;
                        overflow: hidden;
                    ">
                        <div style="
                            background: linear-gradient(90deg, #6366F1, #8B5CF6);
                            width: {bar_width}%;
                            height: 100%;
                        "></div>
                    </div>
                </div>
                <div style="
                    font-size: 1.25rem;
                    font-weight: 700;
                    color: #F1F5F9;
                    min-width: 4rem;
                    text-align: right;
                ">{win_rate:.1%}</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_win_matrix(win_matrix: dict[str, dict[str, float]], model_names: list[str]) -> None:
    """Render the win rate matrix.

    Args:
        win_matrix: Win rate matrix dict
        model_names: List of model names
    """
    if len(model_names) < 2:
        return

    st.markdown(
        """<div style="font-weight: 600; color: #94A3B8; font-size: 0.85rem; margin: 1rem 0 0.5rem;">
            Win Matrix (Row vs Column)
        </div>""",
        unsafe_allow_html=True,
    )

    # Build table HTML
    header_cells = "".join(
        f'<th style="padding: 0.5rem; color: #94A3B8; font-weight: 500;">{name[:10]}</th>' for name in model_names
    )

    rows_html = ""
    for row_name in model_names:
        cells = ""
        for col_name in model_names:
            if row_name == col_name:
                cells += '<td style="padding: 0.5rem; color: #475569; text-align: center;">—</td>'
            else:
                rate = win_matrix.get(row_name, {}).get(col_name, 0.0)
                color = "#10B981" if rate > 0.5 else "#EF4444" if rate < 0.5 else "#94A3B8"
                cell_style = f"padding: 0.5rem; color: {color}; text-align: center; font-weight: 600;"
                cells += f'<td style="{cell_style}">{rate:.0%}</td>'
        rows_html += f"""
            <tr>
                <td style="padding: 0.5rem; color: #94A3B8; font-weight: 500;">{row_name[:10]}</td>
                {cells}
            </tr>
        """

    st.markdown(
        f"""<div style="overflow-x: auto;">
            <table style="
                width: 100%;
                border-collapse: collapse;
                background: rgba(30, 41, 59, 0.3);
                border-radius: 8px;
                font-size: 0.85rem;
            ">
                <thead>
                    <tr style="border-bottom: 1px solid rgba(100, 116, 139, 0.3);">
                        <th style="padding: 0.5rem; color: #64748B;"></th>
                        {header_cells}
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_summary_stats(result: dict[str, Any]) -> None:
    """Render summary statistics.

    Args:
        result: Evaluation result dict
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{result.get('total_queries', 0)}</div>
                <div class="metric-label">Test Queries</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{result.get('total_comparisons', 0)}</div>
                <div class="metric-label">Comparisons</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        best = result.get("best_pipeline", "N/A")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="font-size: 1rem;">{best[:15]}</div>
                <div class="metric-label">Best Model</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_result_panel(
    result: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> None:
    """Render the result panel showing evaluation results.

    Args:
        result: Evaluation result dictionary
        output_dir: Output directory path for reports
    """
    st.markdown(
        """<div class="section-header">
            <span style="margin-right: 0.5rem;">🏆</span>Evaluation Results
        </div>""",
        unsafe_allow_html=True,
    )

    if not result:
        st.markdown(
            """<div style="
                text-align: center;
                padding: 3rem;
                color: #64748B;
                background: rgba(30, 41, 59, 0.3);
                border-radius: 8px;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 0.9rem;">
                    Results will appear here after evaluation completes
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # Summary stats
    _render_summary_stats(result)

    st.markdown('<div class="custom-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

    # Rankings
    st.markdown(
        """<div style="font-weight: 600; color: #F1F5F9; font-size: 1rem; margin-bottom: 0.75rem;">
            Model Rankings
        </div>""",
        unsafe_allow_html=True,
    )

    rankings = result.get("rankings", [])
    for rank, (name, win_rate) in enumerate(rankings, 1):
        _render_ranking_card(rank, name, win_rate, is_best=rank == 1)

    # Win matrix
    win_matrix = result.get("win_matrix", {})
    model_names = [name for name, _ in rankings]
    if win_matrix and len(model_names) > 1:
        _render_win_matrix(win_matrix, model_names)

    # Report and chart links
    if output_dir:
        output_path = Path(output_dir)

        st.markdown('<div class="custom-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

        report_path = output_path / "evaluation_report.md"
        chart_path = output_path / "win_rate_chart.png"

        # Show chart preview
        if chart_path.exists():
            st.markdown(
                """<div style="font-weight: 600; color: #F1F5F9; font-size: 1rem; margin-bottom: 0.75rem;">
                    📊 Win Rate Chart
                </div>""",
                unsafe_allow_html=True,
            )
            st.image(str(chart_path), use_container_width=True)

        # Show report content
        if report_path.exists():
            st.markdown('<div class="custom-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
            st.markdown(
                """<div style="font-weight: 600; color: #F1F5F9; font-size: 1rem; margin-bottom: 0.75rem;">
                    📄 Evaluation Report
                </div>""",
                unsafe_allow_html=True,
            )

            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()

            # Display report in expandable section
            with st.expander("View Full Report / 查看完整报告", expanded=False):
                st.markdown(report_content)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Download Report (.md)",
                    data=report_content,
                    file_name="evaluation_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col2:
                if chart_path.exists():
                    with open(chart_path, "rb") as f:
                        chart_data = f.read()
                    st.download_button(
                        "📊 Download Chart (.png)",
                        data=chart_data,
                        file_name="win_rate_chart.png",
                        mime="image/png",
                        use_container_width=True,
                    )
        else:
            st.info("📝 Report is being generated... / 报告正在生成中...")
