# -*- coding: utf-8 -*-
"""Progress panel for Auto Arena feature.

Displays real-time progress of the evaluation pipeline stages.
"""

from typing import Any

import streamlit as st
from shared.i18n import get_ui_language

# Stage definitions with display info
EVALUATION_STAGES = [
    {
        "id": "queries",
        "name": "Generating Queries",
        "name_zh": "生成测试查询",
        "icon": "📝",
        "description": "Creating diverse test queries based on task description",
        "description_zh": "根据任务描述创建多样化的测试查询",
    },
    {
        "id": "responses",
        "name": "Collecting Responses",
        "name_zh": "收集模型响应",
        "icon": "🤖",
        "description": "Getting responses from all target models",
        "description_zh": "获取所有目标模型的响应",
    },
    {
        "id": "rubrics",
        "name": "Generating Rubrics",
        "name_zh": "生成评估标准",
        "icon": "📋",
        "description": "Creating evaluation criteria",
        "description_zh": "创建评估标准",
    },
    {
        "id": "evaluation",
        "name": "Running Evaluation",
        "name_zh": "执行成对评估",
        "icon": "⚖️",
        "description": "Comparing model responses pairwise",
        "description_zh": "成对比较模型响应",
    },
    {
        "id": "analysis",
        "name": "Analyzing Results",
        "name_zh": "分析评估结果",
        "icon": "📊",
        "description": "Computing rankings and generating report",
        "description_zh": "计算排名并生成报告",
    },
]


def _get_stage_status(  # pylint: disable=unused-argument,too-many-return-statements
    current_stage: str, stage_id: str, stage_index: int
) -> str:
    """Get the status of a stage.

    Args:
        current_stage: ID of the currently running stage
        stage_id: ID of the stage to check
        stage_index: Index of the stage

    Returns:
        Status string: 'completed', 'running', or 'pending'
    """
    # Handle terminal states - all stages should show as completed
    if current_stage in ["completed", "analysis"]:
        # When completed or at analysis, all stages are done
        if current_stage == "completed":
            return "completed"
        # At analysis stage
        if stage_index < len(EVALUATION_STAGES) - 1:
            return "completed"
        return "running"

    # Handle failed/paused - show progress up to failure point
    if current_stage in ["failed", "paused"]:
        # For failed/paused, we can't determine exact progress
        # so show all as pending (caller should handle this case)
        return "pending"

    # Find current stage index
    current_index = -1
    for i, stage in enumerate(EVALUATION_STAGES):
        if stage["id"] == current_stage:
            current_index = i
            break

    # If current stage not found in list, check if it's a known stage
    if current_index < 0:
        return "pending"

    if stage_index < current_index:
        return "completed"
    elif stage_index == current_index:
        return "running"
    else:
        return "pending"


def _render_stage_item(
    stage: dict[str, Any],
    status: str,
    progress: float = 0.0,
    message: str = "",
) -> None:
    """Render a single stage item.

    Args:
        stage: Stage definition dict
        status: 'completed', 'running', or 'pending'
        progress: Progress percentage (0-1) for running stage
        message: Progress message
    """
    # Status colors and icons
    status_config = {
        "completed": {"color": "#10B981", "bg": "rgba(16, 185, 129, 0.1)", "icon": "✓"},
        "running": {"color": "#6366F1", "bg": "rgba(99, 102, 241, 0.1)", "icon": "●"},
        "pending": {"color": "#64748B", "bg": "rgba(100, 116, 139, 0.05)", "icon": "○"},
    }

    config = status_config.get(status, status_config["pending"])

    # Container style
    container_style = f"""
        background: {config['bg']};
        border-left: 3px solid {config['color']};
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
    """

    # Get localized name and description based on UI language
    is_chinese = get_ui_language() == "zh"
    display_name = stage.get("name_zh", stage["name"]) if is_chinese else stage["name"]
    display_desc = stage.get("description_zh", stage["description"]) if is_chinese else stage["description"]

    st.markdown(
        f"""<div style="{container_style}">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.25rem;">{stage['icon']}</span>
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: #F1F5F9; font-size: 0.9rem;">
                        {display_name}
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">
                        {display_desc}
                    </div>
                </div>
                <span style="color: {config['color']}; font-size: 1rem; font-weight: 600;">
                    {config['icon']}
                </span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Show progress bar for running stage
    if status == "running" and progress > 0:
        pct = min(max(progress * 100, 0), 100)
        st.markdown(
            f"""<div style="margin: -0.25rem 0 0.5rem 2.5rem;">
                <div style="
                    background: rgba(100, 116, 139, 0.2);
                    border-radius: 4px;
                    height: 6px;
                    overflow: hidden;
                ">
                    <div style="
                        background: linear-gradient(90deg, #6366F1, #8B5CF6);
                        width: {pct}%;
                        height: 100%;
                        transition: width 0.3s ease;
                    "></div>
                </div>
                <div style="font-size: 0.7rem; color: #64748B; margin-top: 0.25rem;">
                    {message or f'{pct:.0f}% complete'}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_progress_panel(
    is_running: bool = False,
    current_stage: str = "",
    stage_progress: dict[str, Any] | None = None,
    total_progress: float = 0.0,
    logs: list[str] | None = None,
) -> None:
    """Render the progress panel showing evaluation stages.

    Args:
        is_running: Whether evaluation is currently running
        current_stage: ID of the current stage
        stage_progress: Progress info for each stage
        total_progress: Overall progress (0-1)
        logs: List of log messages
    """
    stage_progress = stage_progress or {}
    logs = logs or []

    st.markdown(
        """<div class="section-header">
            <span style="margin-right: 0.5rem;">📈</span>Evaluation Progress
        </div>""",
        unsafe_allow_html=True,
    )

    if not is_running and not current_stage:
        # Not started state
        st.markdown(
            """<div style="
                text-align: center;
                padding: 2rem;
                color: #64748B;
                background: rgba(30, 41, 59, 0.3);
                border-radius: 8px;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚀</div>
                <div style="font-size: 0.9rem;">
                    Configure your evaluation settings and click<br/>
                    <strong style="color: #6366F1;">Start Evaluation</strong> to begin
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # Overall progress bar
    if is_running or total_progress > 0:
        pct = min(max(total_progress * 100, 0), 100)
        status_text = "Running..." if is_running else ("Completed!" if pct >= 100 else "Paused")
        status_color = "#6366F1" if is_running else ("#10B981" if pct >= 100 else "#F59E0B")

        st.markdown(
            f"""<div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="font-size: 0.8rem; color: #94A3B8;">Overall Progress</span>
                    <span style="font-size: 0.8rem; color: {status_color}; font-weight: 600;">
                        {status_text} {pct:.0f}%
                    </span>
                </div>
                <div style="
                    background: rgba(100, 116, 139, 0.2);
                    border-radius: 6px;
                    height: 10px;
                    overflow: hidden;
                ">
                    <div style="
                        background: linear-gradient(90deg, #6366F1, #8B5CF6);
                        width: {pct}%;
                        height: 100%;
                        transition: width 0.5s ease;
                    "></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Stage items
    for i, stage in enumerate(EVALUATION_STAGES):
        status = _get_stage_status(current_stage, stage["id"], i)
        progress = stage_progress.get(stage["id"], {}).get("progress", 0.0)
        message = stage_progress.get(stage["id"], {}).get("message", "")
        _render_stage_item(stage, status, progress, message)

    # Logs section
    if logs:
        with st.expander("Logs", expanded=False):
            log_text = "\n".join(logs[-50:])  # Show last 50 logs
            st.code(log_text, language="text")
