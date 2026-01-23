# -*- coding: utf-8 -*-
"""Grader feature implementation for OpenJudge Studio.

Supports both single evaluation and batch evaluation modes.
"""

from typing import Any

import streamlit as st
from core.base_feature import BaseFeature
from features.grader.components.batch.batch_history_panel import (
    render_batch_history_panel,
    render_batch_task_detail,
)
from features.grader.components.batch.batch_progress_panel import (
    render_batch_progress_panel,
    render_empty_progress_state,
)
from features.grader.components.batch.batch_result_panel import (
    render_batch_result_panel,
)
from features.grader.components.batch.upload_panel import render_upload_panel
from features.grader.components.input_panel import render_input_panel_with_button
from features.grader.components.result_panel import render_result_panel
from features.grader.components.sidebar import render_grader_sidebar
from features.grader.services.batch_history_manager import BatchHistoryManager
from features.grader.services.batch_runner import (
    BatchProgress,
    BatchRunner,
    BatchStatus,
)
from shared.components.common import render_divider
from shared.utils.helpers import run_async


class GraderFeature(BaseFeature):
    """Grader evaluation feature.

    Provides UI for evaluating LLM responses using OpenJudge's built-in graders.
    Supports both single evaluation and batch evaluation modes.
    """

    feature_id = "grader"
    feature_name = "Grader 评估"
    feature_icon = "⚖️"
    feature_description = "使用内置 Grader 评估数据（支持单条和批量）"
    order = 1

    # Session state keys for batch evaluation
    STATE_BATCH_TASK_ID = "batch_task_id"
    STATE_BATCH_PROGRESS = "batch_progress"
    STATE_BATCH_RESULTS = "batch_results"
    STATE_BATCH_VIEWING_TASK = "batch_viewing_task"
    STATE_CURRENT_TAB = "grader_current_tab"

    def render_sidebar(self) -> dict[str, Any]:
        """Render the grader-specific sidebar configuration.

        Returns:
            Dictionary containing all sidebar configuration values
        """
        # Check if in batch mode from session state
        current_tab = st.session_state.get(self.STATE_CURRENT_TAB, 0)
        batch_mode = current_tab == 1  # Tab index 1 is batch evaluation

        return render_grader_sidebar(batch_mode=batch_mode)

    def render_main_content(self, sidebar_config: dict[str, Any]) -> None:
        """Render the main content area for grader evaluation.

        Args:
            sidebar_config: Configuration from the sidebar
        """
        # Initialize session state
        self._init_session_state()

        # Tab navigation
        tab_single, tab_batch, tab_history, tab_help = st.tabs(
            [
                "🔹 Single Evaluation / 单条评估",
                "📦 Batch Evaluation / 批量评估",
                "📜 History / 历史记录",
                "❓ Help / 帮助",
            ]
        )

        with tab_single:
            # Update current tab in session state
            if st.session_state.get(self.STATE_CURRENT_TAB) != 0:
                st.session_state[self.STATE_CURRENT_TAB] = 0
            self._render_single_evaluation(sidebar_config)

        with tab_batch:
            if st.session_state.get(self.STATE_CURRENT_TAB) != 1:
                st.session_state[self.STATE_CURRENT_TAB] = 1
            self._render_batch_evaluation(sidebar_config)

        with tab_history:
            self._render_history_view(sidebar_config)

        with tab_help:
            self._render_quick_guide()

    def _render_single_evaluation(self, sidebar_config: dict[str, Any]) -> None:
        """Render single evaluation view (original functionality)."""
        render_divider()

        # Two-column layout
        col_input, col_result = st.columns([1, 1], gap="large")

        # Input Column
        with col_input:
            input_data, run_flag = render_input_panel_with_button(sidebar_config)

        # Result Column
        with col_result:
            render_result_panel(sidebar_config, input_data, run_flag)

    def _render_batch_evaluation(self, sidebar_config: dict[str, Any]) -> None:
        """Render batch evaluation view."""
        render_divider()

        # Two-column layout
        col_upload, col_progress = st.columns([1, 1], gap="large")

        # Left column: Upload and configuration
        with col_upload:
            upload_result = render_upload_panel(sidebar_config)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            # Start button
            grader_config = sidebar_config.get("grader_config")
            grader_name = sidebar_config.get("grader_name")
            api_key = sidebar_config.get("api_key", "")
            requires_model = grader_config.get("requires_model", True) if grader_config else True

            can_start = upload_result.get("is_valid", False) and grader_name and (not requires_model or api_key)

            start_clicked = st.button(
                "🚀 Start Batch Evaluation / 开始批量评估",
                type="primary",
                use_container_width=True,
                disabled=not can_start,
            )

            if not can_start:
                missing = []
                if not upload_result.get("is_valid"):
                    missing.append("Valid data file")
                if requires_model and not api_key:
                    missing.append("API Key")
                if not grader_name:
                    missing.append("Grader selection")
                if missing:
                    st.caption(f"Missing: {', '.join(missing)}")

        # Right column: Progress and results
        with col_progress:
            progress = st.session_state.get(self.STATE_BATCH_PROGRESS)
            results = st.session_state.get(self.STATE_BATCH_RESULTS)
            task_id = st.session_state.get(self.STATE_BATCH_TASK_ID)

            if results and progress and progress.status == BatchStatus.COMPLETED:
                # Show results
                score_range = grader_config.get("score_range", (0, 1)) if grader_config else (0, 1)
                render_batch_result_panel(
                    task_id=task_id or "",
                    results=results,
                    summary=st.session_state.get("batch_summary"),
                    score_range=tuple(score_range),
                )
            elif progress and progress.status != BatchStatus.PENDING:
                # Show progress
                render_batch_progress_panel(
                    progress=progress,
                    is_running=progress.status == BatchStatus.RUNNING,
                )
            else:
                # Show empty state
                render_empty_progress_state()

        # Handle start button click
        if start_clicked:
            self._start_batch_evaluation(sidebar_config, upload_result, col_progress)

    def _start_batch_evaluation(
        self,
        sidebar_config: dict[str, Any],
        upload_result: dict[str, Any],
        progress_placeholder: Any,
    ) -> None:
        """Start a batch evaluation.

        Args:
            sidebar_config: Sidebar configuration
            upload_result: Upload panel result with parsed data
            progress_placeholder: Streamlit column for progress display
        """
        grader_name = sidebar_config.get("grader_name", "")
        grader_config = sidebar_config.get("grader_config", {})
        data = upload_result.get("parsed_data", [])

        if not data:
            st.error("No data to evaluate")
            return

        # Create history manager and generate task ID
        history_manager = BatchHistoryManager()
        task_id = history_manager.generate_task_id()

        # Create API config
        api_config = {
            "api_endpoint": sidebar_config.get("api_endpoint"),
            "api_key": sidebar_config.get("api_key"),
            "model_name": sidebar_config.get("model_name"),
            "threshold": sidebar_config.get("threshold", 0.5),
            "language": sidebar_config.get("language"),
            "extra_params": sidebar_config.get("extra_params", {}),
        }

        # Create and run batch runner
        max_concurrency = sidebar_config.get("max_concurrency", 10)

        with progress_placeholder:
            with st.status("🔄 Running Batch Evaluation...", expanded=True) as status:
                try:
                    st.write(f"**Task ID:** {task_id}")
                    st.write(f"**Grader:** {grader_name}")
                    st.write(f"**Data count:** {len(data)}")
                    st.write(f"**Max concurrency:** {max_concurrency}")
                    st.write("---")

                    # Create a placeholder for progress updates
                    progress_text = st.empty()
                    progress_bar = st.empty()

                    # Progress callback to update UI during evaluation
                    last_update_count = [0]  # Use list to allow modification in closure

                    def on_progress(prog: BatchProgress) -> None:
                        # Only update every 5 items or at completion to avoid too many updates
                        if prog.completed_count - last_update_count[0] >= 5 or prog.completed_count == prog.total_count:
                            last_update_count[0] = prog.completed_count
                            pct = (prog.completed_count / prog.total_count * 100) if prog.total_count > 0 else 0
                            progress_text.write(
                                f"📊 Progress: **{prog.completed_count}/{prog.total_count}** "
                                f"(✓ {prog.success_count} success, ✗ {prog.failed_count} failed)"
                            )
                            progress_bar.progress(pct / 100, text=f"{pct:.0f}%")

                    # Create runner with progress callback
                    runner = BatchRunner(
                        task_id=task_id,
                        grader_name=grader_name,
                        grader_config=grader_config,
                        api_config=api_config,
                        data=data,
                        max_concurrency=max_concurrency,
                        history_manager=history_manager,
                        progress_callback=on_progress,
                    )

                    # Run evaluation
                    st.write("Starting evaluation...")
                    progress = run_async(runner.run())

                    # Clear progress placeholders
                    progress_text.empty()
                    progress_bar.empty()

                    # Store results in session state
                    st.session_state[self.STATE_BATCH_TASK_ID] = task_id
                    st.session_state[self.STATE_BATCH_PROGRESS] = progress
                    st.session_state[self.STATE_BATCH_RESULTS] = runner.get_results()
                    st.session_state["batch_summary"] = runner.get_summary()

                    # Update status
                    if progress.status == BatchStatus.COMPLETED:
                        complete_label = (
                            f"✅ Evaluation Complete! "
                            f"({progress.success_count} success, "
                            f"{progress.failed_count} failed)"
                        )
                        status.update(label=complete_label, state="complete")
                        st.write("---")
                        st.write(f"✅ **Completed:** {progress.completed_count}/{progress.total_count}")
                        st.write(f"✓ **Success:** {progress.success_count}")
                        st.write(f"✗ **Failed:** {progress.failed_count}")

                        summary = runner.get_summary()
                        if summary.get("avg_score") is not None:
                            st.write(f"📊 **Average Score:** {summary['avg_score']:.2f}")
                        if summary.get("pass_rate") is not None:
                            st.write(f"📈 **Pass Rate:** {summary['pass_rate'] * 100:.1f}%")
                    else:
                        status.update(
                            label=f"⚠️ Evaluation {progress.status.value}",
                            state="error" if progress.status == BatchStatus.FAILED else "complete",
                        )

                except Exception as e:
                    status.update(label="❌ Evaluation Failed", state="error")
                    st.error(f"Error: {e}")
                    st.write("---")
                    st.write("💡 **Tip:** Check the History tab to see partial results or resume.")

    def _render_history_view(self, sidebar_config: dict[str, Any]) -> None:
        """Render the history view."""
        viewing_task = st.session_state.get(self.STATE_BATCH_VIEWING_TASK)

        if viewing_task:
            # Show task detail
            render_batch_task_detail(
                task_id=viewing_task,
                on_back=self._on_back_from_detail,
            )
        else:
            # Check if API key is configured for resume functionality
            api_key = sidebar_config.get("api_key", "")
            if not api_key:
                st.warning(
                    "⚠️ To resume a task, please configure your API Key in the sidebar first.\n\n"
                    "要续传任务，请先在侧边栏配置 API Key。"
                )

            # Show history list with resume callback that has access to sidebar_config
            render_batch_history_panel(
                on_view=self._on_view_task,
                on_resume=lambda task_id: self._on_resume_task(task_id, sidebar_config),
                on_delete=self._on_delete_task,
                limit=20,
            )

    def _on_view_task(self, task_id: str) -> None:
        """Handle view task button click."""
        st.session_state[self.STATE_BATCH_VIEWING_TASK] = task_id
        st.rerun()

    def _on_resume_task(self, task_id: str, sidebar_config: dict[str, Any]) -> None:
        """Handle resume task button click.

        Args:
            task_id: Task ID to resume
            sidebar_config: Current sidebar configuration with API credentials
        """
        # Validate API key is available
        api_key = sidebar_config.get("api_key", "")
        if not api_key:
            st.error(
                "❌ API Key is required to resume evaluation. "
                "Please configure it in the sidebar.\n\n"
                "需要 API Key 才能续传评估，请在侧边栏配置。"
            )
            return

        st.info(f"Resuming task: {task_id}...")

        # Build api_config from current sidebar settings
        api_config = {
            "api_endpoint": sidebar_config.get("api_endpoint"),
            "api_key": api_key,
            "model_name": sidebar_config.get("model_name"),
            "threshold": sidebar_config.get("threshold", 0.5),
            "language": sidebar_config.get("language"),
            "extra_params": sidebar_config.get("extra_params", {}),
        }

        # Load task and resume with current API config
        runner = BatchRunner.resume(task_id, api_config=api_config)
        if runner is None:
            st.error("Failed to resume task. Checkpoint may be corrupted.")
            return

        # Run and show progress
        with st.status("🔄 Resuming Batch Evaluation...", expanded=True) as status:
            try:
                progress = run_async(runner.run())

                st.session_state[self.STATE_BATCH_TASK_ID] = task_id
                st.session_state[self.STATE_BATCH_PROGRESS] = progress
                st.session_state[self.STATE_BATCH_RESULTS] = runner.get_results()
                st.session_state["batch_summary"] = runner.get_summary()

                if progress.status == BatchStatus.COMPLETED:
                    status.update(label="✅ Resume Complete!", state="complete")
                else:
                    status.update(label=f"⚠️ {progress.status.value}", state="error")

            except Exception as e:
                status.update(label="❌ Resume Failed", state="error")
                st.error(f"Error: {e}")

    def _on_delete_task(self, task_id: str) -> None:
        """Handle delete task button click."""
        history_manager = BatchHistoryManager()
        if history_manager.delete_task(task_id):
            st.success(f"Task {task_id} deleted")
            st.rerun()
        else:
            st.error("Failed to delete task")

    def _on_back_from_detail(self) -> None:
        """Handle back button from task detail."""
        st.session_state[self.STATE_BATCH_VIEWING_TASK] = None
        st.rerun()

    def _init_session_state(self) -> None:
        """Initialize session state variables."""
        if "evaluation_result" not in st.session_state:
            st.session_state.evaluation_result = None
        if self.STATE_BATCH_PROGRESS not in st.session_state:
            st.session_state[self.STATE_BATCH_PROGRESS] = None
        if self.STATE_BATCH_RESULTS not in st.session_state:
            st.session_state[self.STATE_BATCH_RESULTS] = None
        if self.STATE_BATCH_TASK_ID not in st.session_state:
            st.session_state[self.STATE_BATCH_TASK_ID] = None
        if self.STATE_BATCH_VIEWING_TASK not in st.session_state:
            st.session_state[self.STATE_BATCH_VIEWING_TASK] = None
        if self.STATE_CURRENT_TAB not in st.session_state:
            st.session_state[self.STATE_CURRENT_TAB] = 0

    def _render_quick_guide(self) -> None:
        """Render the quick start guide."""
        st.markdown(
            """<div class="feature-card">
<div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem;">
    Quick Start Guide / 快速入门
</div>

<div style="margin-bottom: 1.5rem;">
    <div style="color: #A5B4FC; font-weight: 500; margin-bottom: 0.5rem;">
        🔹 Single Evaluation / 单条评估
    </div>
    <div class="guide-step">
        <div class="guide-number">1</div>
        <div class="guide-text">
            Configure API endpoint and key in sidebar
            <br/><span style="color: #64748B;">在侧边栏配置 API 端点和密钥</span>
        </div>
    </div>
    <div class="guide-step">
        <div class="guide-number">2</div>
        <div class="guide-text">
            Select grader category and specific grader
            <br/><span style="color: #64748B;">选择评估器类别和具体评估器</span>
        </div>
    </div>
    <div class="guide-step">
        <div class="guide-number">3</div>
        <div class="guide-text">
            Enter evaluation data (query, response, etc.)
            <br/><span style="color: #64748B;">输入评估数据（问题、回答等）</span>
        </div>
    </div>
    <div class="guide-step">
        <div class="guide-number">4</div>
        <div class="guide-text">
            Click "Run Evaluation" to see results
            <br/><span style="color: #64748B;">点击"运行评估"查看结果</span>
        </div>
    </div>
</div>

<div>
    <div style="color: #A5B4FC; font-weight: 500; margin-bottom: 0.5rem;">
        📦 Batch Evaluation / 批量评估
    </div>
    <div class="guide-step">
        <div class="guide-number">1</div>
        <div class="guide-text">
            Configure API and select grader (same as single)
            <br/><span style="color: #64748B;">配置 API 并选择评估器（同上）</span>
        </div>
    </div>
    <div class="guide-step">
        <div class="guide-number">2</div>
        <div class="guide-text">
            Upload JSON or CSV file with evaluation data
            <br/><span style="color: #64748B;">上传包含评估数据的 JSON 或 CSV 文件</span>
        </div>
    </div>
    <div class="guide-step">
        <div class="guide-number">3</div>
        <div class="guide-text">
            Click "Start Batch Evaluation"
            <br/><span style="color: #64748B;">点击"开始批量评估"</span>
        </div>
    </div>
    <div class="guide-step">
        <div class="guide-number">4</div>
        <div class="guide-text">
            View results and export (supports resume if interrupted)
            <br/><span style="color: #64748B;">查看结果并导出（支持断点续传）</span>
        </div>
    </div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

        # Data format guide
        pre_style = "background: rgba(30,41,59,0.8); padding: 0.5rem; border-radius: 4px; overflow-x: auto;"
        st.markdown(
            f"""<div class="feature-card" style="margin-top: 1rem;">
<div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem;">
    📋 Data Format Guide / 数据格式说明
</div>
<div style="color: #94A3B8; font-size: 0.85rem;">
    <p><strong>JSON Format / JSON 格式:</strong></p>
    <pre style="{pre_style}">{{
  "data": [
    {{
      "query": "User question",
      "response": "Model response",
      "reference_response": "Expected answer (optional)"
    }}
  ]
}}</pre>
    <p style="margin-top: 1rem;"><strong>CSV Format / CSV 格式:</strong></p>
    <pre style="{pre_style}">query,response,reference_response
"Question 1","Answer 1","Reference 1"
"Question 2","Answer 2",""</pre>
    <p style="margin-top: 1rem; color: #FCD34D;">
        ⚠️ Note: Agent graders require JSON format for complex fields.
        <br/>注意：Agent 评估器的复杂字段需要使用 JSON 格式。
    </p>
    <p style="color: #FCD34D;">
        ⚠️ Multimodal graders are not supported for batch evaluation.
        <br/>多模态评估器不支持批量评估。
    </p>
</div>
</div>""",
            unsafe_allow_html=True,
        )

    def on_mount(self) -> None:
        """Initialize grader feature state when mounted."""
        self._init_session_state()
