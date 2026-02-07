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
from shared.i18n import t
from shared.utils.helpers import run_async


class GraderFeature(BaseFeature):
    """Grader evaluation feature.

    Provides UI for evaluating LLM responses using OpenJudge's built-in graders.
    Supports both single evaluation and batch evaluation modes.
    """

    feature_id = "grader"
    feature_name = "Grader Evaluation"  # Default/fallback name
    feature_icon = "⚖️"
    feature_description = "Evaluate data using built-in Graders"  # Default/fallback
    order = 1

    @property
    def display_label(self) -> str:
        """Get the display label for navigation with i18n support."""
        return f"{self.feature_icon} {t('grader.name')}"

    def render_header(self) -> None:
        """Render the feature header with i18n support."""
        st.markdown(
            f"""<div style="margin-bottom: 1rem;">
                <h1 style="
                    font-size: 2rem;
                    font-weight: 700;
                    color: #F1F5F9;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                ">
                    <span>{self.feature_icon}</span>
                    <span>{t('grader.name')}</span>
                </h1>
                <p style="color: #94A3B8; margin: 0.25rem 0 0 0; font-size: 0.95rem;">
                    {t('grader.description')}
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

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
                f"🔹 {t('grader.tabs.single')}",
                f"📦 {t('grader.tabs.batch')}",
                f"📜 {t('grader.tabs.history')}",
                f"❓ {t('grader.tabs.help')}",
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
                f"🚀 {t('grader.batch.start')}",
                type="primary",
                use_container_width=True,
                disabled=not can_start,
            )

            if not can_start:
                missing = []
                if not upload_result.get("is_valid"):
                    missing.append(t("grader.batch.missing_data"))
                if requires_model and not api_key:
                    missing.append(t("grader.batch.missing_api_key"))
                if not grader_name:
                    missing.append(t("grader.batch.missing_grader"))
                if missing:
                    st.caption(f"{t('grader.input.missing')}: {', '.join(missing)}")

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
            st.error(t("grader.error.no_data"))
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
            with st.status(f"🔄 {t('grader.batch.running')}", expanded=True) as status:
                try:
                    st.write(f"**{t('grader.batch.task_id')}:** {task_id}")
                    st.write(f"**{t('grader.batch.grader')}:** {grader_name}")
                    st.write(f"**{t('grader.batch.data_count')}:** {len(data)}")
                    st.write(f"**{t('grader.batch.max_concurrency')}:** {max_concurrency}")
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
                            progress_msg = t(
                                "grader.batch.progress_text",
                                completed=prog.completed_count,
                                total=prog.total_count,
                            )
                            success_msg = t("grader.batch.progress_success", count=prog.success_count)
                            failed_msg = t("grader.batch.progress_failed", count=prog.failed_count)
                            progress_text.write(f"📊 {progress_msg} (✓ {success_msg}, ✗ {failed_msg})")
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
                    st.write(t("grader.batch.starting"))
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
                            f"✅ {t('grader.batch.complete')} "
                            f"({t('grader.batch.progress_success', count=progress.success_count)}, "
                            f"{t('grader.batch.progress_failed', count=progress.failed_count)})"
                        )
                        status.update(label=complete_label, state="complete")
                        st.write("---")
                        completed_label = t("grader.batch.completed_count")
                        st.write(f"✅ **{completed_label}:** " f"{progress.completed_count}/{progress.total_count}")
                        st.write(f"✓ **{t('grader.batch.success_count')}:** {progress.success_count}")
                        st.write(f"✗ **{t('grader.batch.failed_count')}:** {progress.failed_count}")

                        summary = runner.get_summary()
                        if summary.get("avg_score") is not None:
                            st.write(f"📊 **{t('grader.batch.avg_score')}:** {summary['avg_score']:.2f}")
                        if summary.get("pass_rate") is not None:
                            st.write(f"📈 **{t('grader.batch.pass_rate')}:** {summary['pass_rate'] * 100:.1f}%")
                    else:
                        status.update(
                            label=f"⚠️ {t('status.error')}: {progress.status.value}",
                            state="error" if progress.status == BatchStatus.FAILED else "complete",
                        )

                except Exception as e:
                    status.update(label=f"❌ {t('grader.batch.failed_status')}", state="error")
                    st.error(t("grader.error.evaluation_failed", error=str(e)))
                    st.write("---")
                    st.write(f"💡 **{t('grader.batch.resume_tip')}")

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
                st.warning(f"⚠️ {t('grader.batch.resume_warning')}")

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
            st.error(f"❌ {t('grader.batch.resume_api_required')}")
            return

        st.info(t("grader.batch.resuming", task_id=task_id))

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
        with st.status(f"🔄 {t('grader.batch.resuming_status')}", expanded=True) as status:
            try:
                progress = run_async(runner.run())

                st.session_state[self.STATE_BATCH_TASK_ID] = task_id
                st.session_state[self.STATE_BATCH_PROGRESS] = progress
                st.session_state[self.STATE_BATCH_RESULTS] = runner.get_results()
                st.session_state["batch_summary"] = runner.get_summary()

                if progress.status == BatchStatus.COMPLETED:
                    status.update(label=f"✅ {t('grader.batch.resume_complete')}", state="complete")
                else:
                    status.update(label=f"⚠️ {progress.status.value}", state="error")

            except Exception as e:
                status.update(label=f"❌ {t('grader.batch.resume_failed')}", state="error")
                st.error(t("grader.error.evaluation_failed", error=str(e)))

    def _on_delete_task(self, task_id: str) -> None:
        """Handle delete task button click."""
        history_manager = BatchHistoryManager()
        if history_manager.delete_task(task_id):
            st.success(t("grader.batch.task_deleted", task_id=task_id))
            st.rerun()
        else:
            st.error(t("grader.batch.delete_failed"))

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
            f"""<div class="feature-card">
<div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem;">
    {t("grader.help.title")}
</div>

<div style="margin-bottom: 1.5rem;">
    <div style="color: #A5B4FC; font-weight: 500; margin-bottom: 0.5rem;">
        🔹 {t("grader.help.single_title")}
    </div>
    <div class="guide-step">
        <div class="guide-number">1</div>
        <div class="guide-text">{t("grader.help.single_step1")}</div>
    </div>
    <div class="guide-step">
        <div class="guide-number">2</div>
        <div class="guide-text">{t("grader.help.single_step2")}</div>
    </div>
    <div class="guide-step">
        <div class="guide-number">3</div>
        <div class="guide-text">{t("grader.help.single_step3")}</div>
    </div>
    <div class="guide-step">
        <div class="guide-number">4</div>
        <div class="guide-text">{t("grader.help.single_step4")}</div>
    </div>
</div>

<div>
    <div style="color: #A5B4FC; font-weight: 500; margin-bottom: 0.5rem;">
        📦 {t("grader.help.batch_title")}
    </div>
    <div class="guide-step">
        <div class="guide-number">1</div>
        <div class="guide-text">{t("grader.help.batch_step1")}</div>
    </div>
    <div class="guide-step">
        <div class="guide-number">2</div>
        <div class="guide-text">{t("grader.help.batch_step2")}</div>
    </div>
    <div class="guide-step">
        <div class="guide-number">3</div>
        <div class="guide-text">{t("grader.help.batch_step3")}</div>
    </div>
    <div class="guide-step">
        <div class="guide-number">4</div>
        <div class="guide-text">{t("grader.help.batch_step4")}</div>
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
    📋 {t("grader.help.format_title")}
</div>
<div style="color: #94A3B8; font-size: 0.85rem;">
    <p><strong>{t("grader.help.json_format")}:</strong></p>
    <pre style="{pre_style}">{{
  "data": [
    {{
      "query": "User question",
      "response": "Model response",
      "reference_response": "Expected answer (optional)"
    }}
  ]
}}</pre>
    <p style="margin-top: 1rem;"><strong>{t("grader.help.csv_format")}:</strong></p>
    <pre style="{pre_style}">query,response,reference_response
"Question 1","Answer 1","Reference 1"
"Question 2","Answer 2",""</pre>
    <p style="margin-top: 1rem; color: #FCD34D;">
        ⚠️ {t("grader.help.agent_note")}
    </p>
    <p style="color: #FCD34D;">
        ⚠️ {t("grader.help.multimodal_note")}
    </p>
</div>
</div>""",
            unsafe_allow_html=True,
        )

    def on_mount(self) -> None:
        """Initialize grader feature state when mounted."""
        self._init_session_state()
